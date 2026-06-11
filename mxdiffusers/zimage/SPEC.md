# Z-Image (`MXZimagePipeline`) — implementation spec

Clean-room transformer target for the second mxdiffusers family. **Source-grounded** against the
official 🤗 diffusers reference (`huggingface/diffusers@main`:
`src/diffusers/models/transformers/transformer_z_image.py`,
`src/diffusers/pipelines/z_image/pipeline_z_image.py`) + the local checkpoint
`Tongyi-MAI/Z-Image-Turbo` (Apache-2.0). The transformer implementation is independent of
mflux; the shared Qwen3/VAE helpers were re-derived from the references on 2026-06-11, so the path is
described as mixed until those helpers are re-derived or split.

## Architecture family

`ZImageTransformer2DModel` is a **NextDiT / Lumina-2-style single-stream DiT** — the weight-key
signature is unmistakable (`cap_embedder`, `context_refiner`, `noise_refiner`, `layers`,
`adaLN_modulation`, `feed_forward.w1/w2/w3`, `x_pad_token`/`cap_pad_token`, 3D `axes_dims`).
Not a dual-stream MMDiT like FLUX.

## Config (Z-Image-Turbo)

| transformer | value | | other | value |
|---|---|---|---|---|
| dim | 3840 | | in/out channels | 16 |
| n_heads / n_kv_heads | 30 / 30 (MHA, head_dim 128) | | patch_size | 2 (f_patch 1) |
| n_layers | 30 | | axes_dims | [32, 48, 48] (Σ=128) |
| n_refiner_layers | 2 (×context, ×noise) | | axes_lens | [1536, 512, 512] |
| qk_norm / norm_eps | true / 1e-5 | | rope_theta | 256.0 |
| cap_feat_dim | 2560 (Qwen3 hidden) | | t_scale | 1000.0 |

- **scheduler**: `FlowMatchEulerDiscreteScheduler`, `use_dynamic_shifting=false`, **shift 3.0**, 1000 train steps.
- **text encoder**: Qwen3 (hidden 2560, 36 layers, 32 heads / 8 KV, head_dim 128, int 9728, θ 1e6).
- **vae**: stock `AutoencoderKL`, flux-dev (latent 16ch, block_out [128,256,512,512], 2 layers/block, `scaling_factor 0.3611`, `shift_factor 0.1159`, no quant/post-quant conv).

## Transformer forward (the new build)

1. **Embed.** Image latents → patchify 2×2 → `all_x_embedder["2-1"]` = `Linear(2·2·16 → 3840)`.
   Caption (Qwen3 hidden 2560) → `cap_embedder` = `RMSNorm(2560) → Linear(2560→3840)`. Learnable
   `x_pad_token`/`cap_pad_token` (1,dim) fill padded positions (`where(mask, pad, feats)`).
2. **Block** (`ZImageTransformerBlock`, sandwich-norm): RMSNorm `attention_norm1/2`, `ffn_norm1/2`;
   attention with **qk-norm** (RMSNorm `norm_q`/`norm_k` on head_dim) + complex RoPE; SwiGLU
   `w2(silu(w1·x)·w3·x)`, `hidden_dim = int(dim/3*8)`.
   - **Non-modulated** (`context_refiner`): `x += attention_norm2(attn(attention_norm1(x)))`;
     `x += ffn_norm2(ff(ffn_norm1(x)))`.
   - **Modulated** (`noise_refiner`, `layers`): `adaLN_modulation = Linear(min(dim,ADALN), 4·dim)`
     → chunk → `(scale_msa, gate_msa, scale_mlp, gate_mlp)`, `scale = 1+scale` (note: **scale+gate
     only, no shift**). `x += gate_msa · attention_norm2(attn(attention_norm1(x)·scale_msa))`;
     `x += gate_mlp · ffn_norm2(ff(ffn_norm1(x)·scale_mlp))`.
3. **Refiners** run before the join: `context_refiner`(×2, no mod) on caption tokens;
   `noise_refiner`(×2, mod) on image tokens.
4. **Sequence assembly:** unified = `[image_tokens, caption_tokens]` per sample (image first),
   right-padded; bool attention mask `(B, seqlen)` → `(B,1,1,seqlen)`.
5. **3D RoPE:** per axis `d∈{32,48,48}`: `freqs = 1/θ^(arange(0,d,2)/d)`, `outer(pos, freqs)`,
   `polar→complex64`. Caption coords `(idx, 0, 0)`; image coords `(offset, h, w)`. Apply via
   `view_as_complex(x)·freqs → view_as_real`.
6. **Timestep:** sinusoidal embed (freq_size 256, `max_period 10000`, `cos‖sin`) → MLP(256→mid→dim);
   `adaln_input = t_embedder(t · 1000)`.
7. **Final:** `LayerNorm(no-affine, eps 1e-6) · (1 + adaLN(c))` → `Linear(dim→out)`; unpatchify
   `(seq, patch) → (16, H, W)`.

### Transformer state_dict templates (ground truth, 521 keys)
`all_x_embedder.2-1.{weight,bias}`, `cap_embedder.{0,1}.{weight,bias}`, `x_pad_token`,
`cap_pad_token`, `t_embedder.mlp.{0,2}.{weight,bias}`,
`{context_refiner,noise_refiner,layers}.N.attention.{to_q,to_k,to_v,to_out.0,norm_q,norm_k}`,
`*.attention_norm{1,2}`, `*.ffn_norm{1,2}`, `*.feed_forward.{w1,w2,w3}`,
`{noise_refiner,layers,all_final_layer.2-1}.*.adaLN_modulation.N`, `all_final_layer.2-1.linear`.

## Pipeline (Turbo)

- **encode_prompt:** Qwen2 chat template (`add_generation_prompt=True`, `enable_thinking=True`),
  `max_sequence_length=512`; `text_encoder(...).hidden_states[-2]` (**second-to-last**, dim 2560);
  attention-masked, **no pooling** (variable-length per sample).
- **latents:** 16ch, `randn`, spatial = `2·(px // (vae_scale_factor·2))` with `vae_scale_factor=8`.
- **denoise:** Turbo is **distilled / CFG-internalized → guidance-free** (8 steps; the base
  pipeline's true-CFG path is unused). Sigmas `linspace(1, 1/n, n)`, **static shift 3.0**:
  `σ' = 3σ / (1 + 2σ)`. Euler: `latents += (σ[t+1]−σ[t])·noise`.
- **decode:** `latents = latents/0.3611 + 0.1159` → `AutoencoderKL.decode`.

## Reuse plan

| Component | Plan |
|---|---|
| Qwen3 text encoder | **Reuse** `mxdiffusers/flux/text_encoder.py` (independent, transformers-derived since 2026-06-11); extract `hidden_states[-2]`, dim 2560. |
| Tokenizer | **Reuse** the Qwen2 chat-template tokenizer (`enable_thinking=True`, no layer-stack). |
| Scheduler | **New small variant**: static-shift flow-match Euler (`σ' = 3σ/(1+2σ)`). |
| VAE | Z-Image uses flux-dev `AutoencoderKL` (16ch). Current code reuses the compatible FLUX decoder helper; replace with an independently-derived standard AutoencoderKL decoder if we want wholly clean-room Z-Image provenance. |
| Transformer | **New build** (NextDiT, above). The single-stream block, qk-norm, SwiGLU, RoPE-application, and timestep embed transfer in spirit from `mxdiffusers/flux/transformer.py`; 3D-RoPE construction, refiners, pad-tokens, and sequence packing are new. |
| Loading | `mxalloy.load_quantized` + a `remap_zimage_*` map (diffusers keys → our modules). |

## Build order (#46)
1. NextDiT transformer module + 3D RoPE + sequence packing.
2. Static-shift scheduler.
3. VAE (reuse `Flux2VAE` if compatible, else AutoencoderKL decoder).
4. Qwen3 encoder reuse (`hidden_states[-2]`) + tokenizer.
5. Latent prep + weight remap.
6. `MXZimagePipeline(MXPipeline)` + register in `surface/server.py` PIPELINES + `MODEL_REGISTRY`.
7. **Verify**: parity oracle is diffusers (not installed; pulls torch). Either install
   diffusers+torch in the dev venv as a dev-only oracle, or validate by generation quality on a
   fixed seed. Clean-room transformer: implement from this spec and keep shared-helper
   provenance visible until those helpers are replaced.
