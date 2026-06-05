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

## Z-Image (`mxdiffusers/zimage`)

The Z-Image transformer was implemented clean-room against the official `diffusers`
`ZImagePipeline` / `ZImageTransformer2DModel` reference and the
`Tongyi-MAI/Z-Image-Turbo` checkpoint (Apache-2.0).

Current caveat: `mxdiffusers/zimage` reuses the shared Qwen3 text encoder and VAE decoder helper
from `mxdiffusers/flux`, and those helpers still carry the FLUX/mflux port lineage described
above. Until those helpers are re-derived or moved to independent shared modules, describe
Z-Image as "clean-room transformer; shared FLUX-derived helpers", not as wholly
mflux-independent.
