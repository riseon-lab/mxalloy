"""MXFluxPipeline — the FLUX family on the mxalloy runtime.

Wraps the resident FLUX.2-klein engine in the diffusers-style ``MXPipeline`` surface. The
model internals live in this package (see ``PROVENANCE.md``); loading + quantization go through
mxalloy. INTERNAL until the API stabilises; requires mlx.
"""

from __future__ import annotations

from mxdiffusers.flux.engine import Flux2KleinEngine
from mxdiffusers.pipeline import MXPipeline, MXResult, OnStep


class MXFluxPipeline(MXPipeline):
    """FLUX.2-klein text-to-image (few-step flow-match + tiled VAE decode)."""

    family = "flux"

    def __init__(self, engine: Flux2KleinEngine) -> None:
        self._engine = engine

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | None = None,
        *,
        quantize_bits: int | None = 4,
        vae_tile_latent: int | None = 128,
        **kwargs,
    ) -> MXFluxPipeline:
        # ``model_id`` is a local checkpoint directory, or None to resolve the cached
        # FLUX.2-klein snapshot. (A general HF-id resolver lands with the adaptive loader.)
        engine = Flux2KleinEngine(
            model_dir=model_id, quantize_bits=quantize_bits, vae_tile_latent=vae_tile_latent
        )
        return cls(engine)

    def __call__(
        self,
        prompt: str,
        *,
        seed: int = 0,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 1.0,
        on_step: OnStep | None = None,
        **kwargs,
    ) -> MXResult:
        image = self._engine.generate(
            prompt,
            seed=seed,
            steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            on_step=on_step,
        )
        return MXResult(images=[image], seed=seed)

    def load_lora_weights(self, path: str, *, scale: float = 1.0) -> dict:
        return self._engine.set_loras([(str(path), float(scale))])

    def unload_lora_weights(self) -> None:
        self._engine.clear_loras()
