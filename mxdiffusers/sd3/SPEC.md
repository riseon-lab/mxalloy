# SD3 / SD3.5 (StableDiffusion3Pipeline) — implementation spec

**Status: planned; blocked on weights access.** `stabilityai/stable-diffusion-3.5-medium`
(and -large) are gated behind the Stability Community License — the dev machine's HF token
returned 403 on 2026-06-11, so the config facts below are from the public architecture
record and must be re-verified against the real checkpoint once access is accepted.
`MXAutoPipeline` recognizes `StableDiffusion3Pipeline` and reports this status.

## Architecture (public record — verify against checkpoint before implementing)

- MMDiT transformer (`SD3Transformer2DModel`): joint image/text attention with separate
  modulation per stream; 3.5-Medium ≈ 2.5 B (24 layers), 3.5-Large ≈ 8 B (38 layers);
  16-ch latents, 2×2 patching.
- Triple text encoding: CLIP-L (768) + CLIP-G/bigG (1280) penultimate + pooled (concat →
  2048 pooled projection), plus T5-XXL sequence context. `text_encoder_3=None` operation
  (drop T5) is a supported degraded mode in the reference — useful on small machines.
- VAE: 16-ch AutoencoderKL (same family as FLUX/SDXL — the planned shared
  `vae_kl.py` covers it).
- Scheduler: FlowMatchEuler with shift 3.0 (3.5 uses dynamic shifting).

## Reuse from what already ships

CLIP-L + bigG: `sdxl/clip.py` (verified). T5: shared module from the FLUX.1 plan.
VAE: shared parameterized KL decoder. Net-new work is the MMDiT graph + weight mapping +
pipeline — the smallest increment of the planned architectures once FLUX.1 lands T5.

## Unblock step (owner)

Accept the Stability Community License for stable-diffusion-3.5-medium on Hugging Face with
the project's account; then `huggingface-cli download stabilityai/stable-diffusion-3.5-medium`.
Note the license's revenue clause before shipping it as a headline feature.
