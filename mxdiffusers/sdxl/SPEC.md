# SDXL (StableDiffusionXLPipeline) — implementation spec

**Source-grounded** against the official 🤗 diffusers reference (`StableDiffusionXLPipeline`,
`UNet2DConditionModel`, `AutoencoderKL`, `EulerDiscreteScheduler`; Apache-2.0 — attributed in
`NOTICE`) and the `stabilityai/stable-diffusion-xl-base-1.0` checkpoint configs. Like
`zimage/`, module attribute names mirror the diffusers state_dict so the weight remap is
near-identity and pure-string testable.

Covers the architecture, not one model: SDXL Base, SDXL Turbo, and SDXL finetunes share this
graph (Turbo differs only in scheduler spacing/steps and guidance).

## Components (from the checkpoint configs)

| component | class | config facts |
|---|---|---|
| text_encoder | CLIPTextModel (CLIP-L) | 12 layers, hidden 768, heads 12, mlp 3072, act `quick_gelu`, max_pos 77, causal |
| text_encoder_2 | CLIPTextModelWithProjection (OpenCLIP bigG) | 32 layers, hidden 1280, heads 20, mlp 5120, act `gelu`, projection 1280, causal |
| unet | UNet2DConditionModel | ch [320,640,1280]; down [Down, XAttnDown, XAttnDown]; up [XAttnUp, XAttnUp, Up]; 2 layers/block; transformer layers/block [1,2,10]; heads [5,10,20] (64-dim heads — `attention_head_dim` is the *head count* when `num_attention_heads` is null); cross-attn dim 2048; `addition_embed_type=text_time`, add_time_embed 256, class-emb input 2816; GN(32); `use_linear_projection` |
| vae (decode only) | AutoencoderKL | latent 4ch, ch [128,256,512,512], 3 resnets/up-block, mid attn (single-head 512), scaling 0.13025 |
| scheduler | EulerDiscreteScheduler | scaled_linear β ∈ [0.00085, 0.012], 1000 train steps, `leading` spacing, offset 1, epsilon pred |

## Conditioning (the SDXL-specific part)

- `prompt_embeds` = concat(CLIP-L `hidden_states[-2]` (768), bigG `hidden_states[-2]` (1280)) → (77, 2048). Penultimate layer, **no** final_layer_norm.
- `pooled` = bigG pooled: final_layer_norm(last_hidden)[argmax(input_ids)] @ text_projection → (1280). (argmax-EOT pooling — both configs carry the historical `eos_token_id=2`, which selects transformers' argmax path; CLIP's real EOT id 49407 is the vocab max, so argmax finds it.)
- `time_ids` = [orig_h, orig_w, crop_top, crop_left, target_h, target_w]; each → sinusoidal(256); flattened (1536); `add_embedding(concat([pooled, time_ids_emb]))` (2816 → 1280) added to the timestep embedding.
- Classifier-free guidance is **required** for base quality (default scale 5.0): unconditional = empty-prompt encode; `eps = eps_u + g·(eps_c − eps_u)`; the cond/uncond pair is batched (batch 2).

## Quantization plan (mxalloy)

Linears quantize 4-bit (all attention/ff dims are multiples of 64); convs/norms stay 16-bit
(`nn.quantize` only touches modules with `to_quantized`). VAE loads from the fp32 shard and is
cast to bf16 at load — fp16 *range* is what NaNs the SDXL VAE, bf16 keeps fp32's exponent.
Text encoders quantize 4-bit; UNet 4-bit ≈ 1.6 GB resident.

## Verification ladder

1. Weight remap: pure-string unit tests (`tests/test_sdxl_weight_mapping.py`).
2. Full coverage: `load_quantized` returns empty missing-set per component.
3. Text encoders: numeric parity vs `transformers` CLIP on the same weights (CPU).
4. Scheduler: sigmas/timesteps parity vs `diffusers.EulerDiscreteScheduler`.
5. End-to-end: same-seed/latents image vs diffusers fp16 on MPS (black-box oracle).

## LoRA

PEFT/diffusers-format SDXL LoRAs (`unet.…lora_A/B`, `text_encoder…`) map via the shared
`mxdiffusers.lora` core; kohya-flattened (`lora_unet_…` underscore) keys are a documented TODO
(needs the known-token unflattening table).
