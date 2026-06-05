"""MXZimagePipeline — the Z-Image family on the mxalloy runtime."""

from __future__ import annotations

from mxdiffusers.pipeline import MXPipeline, MXResult, OnStep
from mxdiffusers.zimage.engine import ZImageEngine


class MXZimagePipeline(MXPipeline):
    """Z-Image-Turbo text-to-image (8-step, guidance-free distilled S3-DiT)."""

    family = "zimage"

    def __init__(self, engine: ZImageEngine) -> None:
        self._engine = engine

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | None = None,
        *,
        quantize_bits: int | None = 4,
        vae_tile_latent: int | None = None,  # accepted for API parity; Z-Image VAE decodes whole
        **kwargs,
    ) -> MXZimagePipeline:
        return cls(ZImageEngine(model_dir=model_id, quantize_bits=quantize_bits))

    def __call__(
        self,
        prompt: str,
        *,
        seed: int = 0,
        num_inference_steps: int = 8,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 0.0,
        cache_threshold: float = 0.25,  # FBC is near-lossless on Z-Image -> on by default
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
            cache_threshold=cache_threshold,
            on_step=on_step,
        )
        return MXResult(images=[image], seed=seed)
