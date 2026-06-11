# Provenance of mxdiffusers model implementations

mxdiffusers is the diffusion framework that runs on the **mxalloy** runtime. Its model-family
adapters implement *published* model architectures so that the official checkpoints load and
produce correct images. This file records where each family's code came from.

A note on what is and isn't anyone's IP: a model's **facts** — its architecture, the
checkpoint's weight-key names, and the published math (flow-match, RoPE, the scheduler shift
constants) — belong to the model's authors and the public record of published algorithms. Any
correct implementation converges on them; that convergence is not copying. Only a specific
implementation's **code expression** is covered by its license.

## FLUX (`mxdiffusers/flux`)

**Re-derived 2026-06-11.** The FLUX.2 family modules (MMDiT transformer, Qwen3 text encoder,
AutoencoderKLFlux2 decode + tiling, flow-match scheduler, latent packing, tokenizer) are an
**independent MLX reimplementation derived from** the Apache-2.0 `diffusers` reference
(`Flux2Transformer2DModel` / `AutoencoderKLFlux2` / `Flux2KleinPipeline`), the `transformers`
Qwen3 reference, and the FLUX.2-klein-4B checkpoint facts. Module attribute names mirror the
checkpoint state_dict (identity remap). No mflux-derived code remains; attribution is carried
in `NOTICE` for diffusers/transformers only.

Verification of the re-derivation:
- per-component key/shape coverage is exact (169 transformer / 398 text-encoder / 142
  decode-side VAE params, zero unmapped/missing/mismatched);
- Qwen3 encoder numeric parity vs `transformers` on the real weights (mean |Δ| ≈ 1e-2 against
  hidden-state std ≈ 76);
- transformer numeric parity vs the diffusers reference on the real weights with identical
  inputs (bf16-class agreement: mean |Δ| 0.018 vs output std 0.545);
- end-to-end generation quality + memory profile preserved (4.54 GB 4-bit load peak), tiled
  decode seam-free at 1536².

History: the previous implementation (see git history before this date) was a close MLX port
of, and verified against, **mflux** (MIT, © Filip Strand). It served as the correctness
baseline this project's benchmarks were built on, and mflux remains the benchmark comparison
target. Releases cut from pre-re-derivation commits must carry mflux's MIT notice.

Note on seed semantics: the re-derived engine follows the diffusers reference noise/schedule
conventions, so a given seed produces a different (equally valid) image than pre-re-derivation
commits did.

## SDXL (`mxdiffusers/sdxl`)

The SDXL family (UNet, dual CLIP text encoders, AutoencoderKL decoder, Euler scheduler) is an
**independent MLX reimplementation derived from** the `diffusers` /`transformers` references
(Apache-2.0, attributed in `NOTICE`) and the `stabilityai/stable-diffusion-xl-base-1.0`
checkpoint configs — source-grounded, same approach as Z-Image (see `sdxl/SPEC.md`). It has
**no mflux lineage** and shares no code with `mxdiffusers/flux`. Verified by per-component
shape coverage, text-encoder numeric parity vs `transformers`, scheduler parity vs
`diffusers`, and a same-latents black-box image comparison against the diffusers pipeline.

## Shared modules (`mxdiffusers/lora.py`, `hub.py`, `auto.py`, `pipeline.py`, `vae_kl.py`)

`lora.py`, `hub.py`, `auto.py`, `pipeline.py`: original mxdiffusers code, no external
lineage. `vae_kl.py` is the shared AutoencoderKL decoder, an independent reimplementation
derived from the diffusers reference (first verified in the SDXL family, then promoted; now
also serves FLUX.2 and Z-Image).

## Z-Image (`mxdiffusers/zimage`)

The Z-Image transformer is an **independent MLX reimplementation derived from** the official
`diffusers` `ZImagePipeline` / `ZImageTransformer2DModel` reference (Apache-2.0, ©
HuggingFace — attributed in `NOTICE`) and the `Tongyi-MAI/Z-Image-Turbo` checkpoint
(Apache-2.0). It was written source-grounded — with the diffusers reference open as the
correctness oracle (see `zimage/SPEC.md`) — so we deliberately do not call it "clean-room";
Apache-2.0 permits this derivation with attribution, which `NOTICE` carries.

The shared helpers Z-Image reuses — the Qwen3 text encoder (`mxdiffusers/flux/text_encoder.py`)
and the KL decoder (`mxdiffusers/vae_kl.py`) — were re-derived from the transformers/diffusers
references on 2026-06-11 (verified by an old-vs-new same-seed Z-Image generation producing the
same image), so the whole Z-Image path is now free of port lineage. The earlier
"shared FLUX-derived helpers" caveat is resolved.
