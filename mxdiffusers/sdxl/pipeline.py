"""MXSDXLPipeline — the SDXL architecture on the mxalloy runtime.

Covers StableDiffusionXLPipeline-class checkpoints: SDXL Base, SDXL Turbo, and SDXL
finetunes that keep the architecture (Turbo wants ``num_inference_steps=1..4, guidance=0``).
INTERNAL until the API stabilises; requires mlx.
"""

from __future__ import annotations

from mxdiffusers.pipeline import MXPipeline, MXResult, OnStep
from mxdiffusers.sdxl.engine import SDXLEngine


class MXSDXLPipeline(MXPipeline):
    """SDXL text-to-image (dual-CLIP conditioning, Euler CFG denoise)."""

    family = "sdxl"

    def __init__(self, engine: SDXLEngine) -> None:
        self._engine = engine

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | None = None,
        *,
        quantize_bits: int | None = 4,
        vae_tile_latent: int | None = None,  # accepted for API parity; SDXL VAE decodes whole
        **kwargs,
    ) -> MXSDXLPipeline:
        return cls(SDXLEngine(model_dir=model_id, quantize_bits=quantize_bits))

    def __call__(
        self,
        prompt: str,
        *,
        seed: int = 0,
        num_inference_steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 5.0,
        negative_prompt: str = "",
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
            negative_prompt=negative_prompt,
            on_step=on_step,
        )
        return MXResult(images=[image], seed=seed)

    def load_lora_weights(self, path: str, *, scale: float = 1.0) -> dict:
        return self._engine.set_loras([(str(path), float(scale))])

    def set_lora_weights(self, loras: list[tuple[str, float]]) -> dict:
        return self._engine.set_loras([(str(path), float(scale)) for path, scale in loras])

    def unload_lora_weights(self) -> None:
        self._engine.clear_loras()
