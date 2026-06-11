# FLUX.1 (FluxPipeline) — implementation spec

**Status: planned, next architecture up.** Config facts below are from the real
`black-forest-labs/FLUX.1-schnell` checkpoint (fetched 2026-06-11); weights are not yet
downloaded on the dev machine (~33 GB). `MXFluxPipeline` (the FLUX family front door) and `MXAutoPipeline` both detect
`FluxPipeline` / `FluxKontextPipeline` checkpoints and report this status.

Covers the architecture: FLUX.1-schnell (Apache-2.0), FLUX.1-dev (non-commercial license,
gated), FLUX.1-Kontext, and finetunes.

## Checkpoint facts (schnell)

- `model_index.json` `_class_name: FluxPipeline`; components: scheduler, text_encoder
  (CLIP-L), text_encoder_2 (T5-XXL), tokenizer ×2, transformer, vae.
- transformer `FluxTransformer2DModel`: 19 double-stream + 38 single-stream blocks,
  24 heads × 128 dim (3072 inner), `joint_attention_dim` 4096 (T5 context),
  `pooled_projection_dim` 768 (CLIP-L pooled), `in_channels` 64 (16-ch latents, 2×2 packed),
  `guidance_embeds: false` for schnell (**true for dev** — config-driven).
- vae `AutoencoderKL`: 16-ch latents, `block_out_channels [128,256,512,512]`,
  `scaling_factor 0.3611`, `shift_factor 0.1159` (decode input: `z/scale + shift`).
- scheduler `FlowMatchEulerDiscreteScheduler`: `shift 1.0`, static for schnell
  (`use_dynamic_shifting: false`; dev uses dynamic base/max shift 0.5/1.15).
- T5-XXL encoder: 24 layers, d_model 4096, 64 heads × d_kv 64, gated `gelu_new` FFN
  (d_ff 10240), relative-attention buckets 32 / max-distance 128, vocab 32128.

## Build plan (lineage-free — no mflux anywhere in this path)

1. `mxdiffusers/t5.py` (shared): T5 encoder stack — RMSNorm, relative position bias,
   gated-gelu FFN. Derived from the transformers reference (Apache-2.0); verify numeric
   parity vs `transformers.T5EncoderModel` exactly as `sdxl/clip.py` was.
2. Reuse `sdxl/clip.py`'s CLIP-L (already verified) for the pooled embedding —
   promote to `mxdiffusers/clip.py` shared module.
3. Parameterize `sdxl/vae.py`'s AutoencoderKL decoder (latent channels, scaling/shift) —
   it is the same family; promote to a shared `mxdiffusers/vae_kl.py`. This *also* gives
   Z-Image an independently-derived decoder, retiring its FLUX-shared-helper caveat.
4. `flux1/transformer.py`: MMDiT double+single stream from the diffusers
   `FluxTransformer2DModel` reference — mirror the state_dict (identity remap), same
   discipline as sdxl/zimage. RoPE axes (16, 56, 56)-style image/text ids as in the
   reference.
5. Flow-match Euler scheduler: `sigma_t = shift·t / (1 + (shift−1)·t)`, published math —
   small standalone module (do not reuse `flux/scheduler.py`, which carries port lineage).
6. Engine/pipeline mirroring `sdxl/engine.py`; quantize transformer+T5 4-bit
   (schnell 12B → ~6.7 GB; T5 4.7B → ~2.6 GB; resident ~10 GB on 18 GB-class machines —
   mflux demonstrates this is viable).

## Verification ladder

Shape coverage from real headers → T5/CLIP parity vs transformers → scheduler unit values →
4-step schnell generation → black-box compare (mflux can serve as a *runtime* oracle —
generating reference images is not code derivation).

## Blockers

- ~33 GB download (disk had ~50 GB free on the dev machine at spec time; clear space or use
  an external volume).
- FLUX.1-dev additionally needs the gated-license acceptance for its weights.
