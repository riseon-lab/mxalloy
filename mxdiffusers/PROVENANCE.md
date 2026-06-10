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

The FLUX.2 family modules (transformer, VAE, Qwen3 text encoder, scheduler, latent packing)
were developed as a close port of, and verified against, **mflux**
(MIT, © Filip Strand — https://github.com/filipstrand/mflux), using mflux as the correctness
oracle while mapping Black Forest Labs' FLUX.2-klein checkpoint.

- There is **no mflux runtime dependency** — mxdiffusers never imports mflux.
- While these modules retain that port lineage, **distributing mxdiffusers requires
  reproducing mflux's MIT license text + copyright notice.**
- Intended direction: re-derive these modules from the FLUX.2 checkpoint + the `diffusers`
  reference so the implementation is independent and carries no port lineage. Once that is
  done the attribution requirement falls away. Until then, treat the FLUX family as
  mflux-derived for attribution purposes.

## SDXL (`mxdiffusers/sdxl`)

The SDXL family (UNet, dual CLIP text encoders, AutoencoderKL decoder, Euler scheduler) is an
**independent MLX reimplementation derived from** the `diffusers` /`transformers` references
(Apache-2.0, attributed in `NOTICE`) and the `stabilityai/stable-diffusion-xl-base-1.0`
checkpoint configs — source-grounded, same approach as Z-Image (see `sdxl/SPEC.md`). It has
**no mflux lineage** and shares no code with `mxdiffusers/flux`. Verified by per-component
shape coverage, text-encoder numeric parity vs `transformers`, scheduler parity vs
`diffusers`, and a same-latents black-box image comparison against the diffusers pipeline.

## Shared modules (`mxdiffusers/lora.py`, `hub.py`, `auto.py`, `pipeline.py`)

Original mxdiffusers code, no external lineage. The LoRA core was hoisted from the (mflux-
attributed) flux family's runtime-LoRA implementation as generic delta-injection machinery;
the per-family key conventions stay in each family's `lora.py`.

## Z-Image (`mxdiffusers/zimage`)

The Z-Image transformer is an **independent MLX reimplementation derived from** the official
`diffusers` `ZImagePipeline` / `ZImageTransformer2DModel` reference (Apache-2.0, ©
HuggingFace — attributed in `NOTICE`) and the `Tongyi-MAI/Z-Image-Turbo` checkpoint
(Apache-2.0). It was written source-grounded — with the diffusers reference open as the
correctness oracle (see `zimage/SPEC.md`) — so we deliberately do not call it "clean-room";
Apache-2.0 permits this derivation with attribution, which `NOTICE` carries.

Current caveat: `mxdiffusers/zimage` reuses the shared Qwen3 text encoder and VAE decoder helper
from `mxdiffusers/flux`, and those helpers still carry the FLUX/mflux port lineage described
above. Until those helpers are re-derived or moved to independent shared modules, describe
Z-Image as "independent transformer; shared FLUX-derived helpers", not as wholly
mflux-independent.
