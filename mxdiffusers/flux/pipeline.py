"""MXFluxPipeline — the FLUX family on the mxalloy runtime.

One pipeline class for the FLUX family; the checkpoint decides the generation. FLUX.2
(klein) runs today on the resident klein engine; FLUX.1 checkpoints (schnell/dev/Kontext,
``FluxPipeline``-class) are detected and report their planned status (see ``FLUX1_SPEC.md``)
instead of failing with a shape error. The model internals live in this package (see
``PROVENANCE.md``); loading + quantization go through mxalloy. INTERNAL until the API
stabilises; requires mlx to run (importing this module stays mlx-free).
"""

from __future__ import annotations

from mxalloy.errors import ModelLoadError
from mxdiffusers.pipeline import MXPipeline, MXResult, OnStep

_FLUX1_CLASSES = {"FluxPipeline", "FluxKontextPipeline"}


class MXFluxPipeline(MXPipeline):
    """FLUX family text-to-image (FLUX.2-klein today; flow-match + tiled VAE decode)."""

    family = "flux"

    def __init__(self, engine) -> None:
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
        # ``model_id``: a local checkpoint directory, an HF repo id (resolved against the
        # local HF cache — never downloaded), or None for the default cached klein snapshot.
        from mxdiffusers.flux.loader import find_klein_model_dir

        model_dir = find_klein_model_dir(model_id)
        cls._reject_unsupported_generation(model_dir)
        from mxdiffusers.flux.engine import Flux2KleinEngine

        engine = Flux2KleinEngine(
            model_dir=model_dir, quantize_bits=quantize_bits, vae_tile_latent=vae_tile_latent
        )
        return cls(engine)

    @staticmethod
    def _reject_unsupported_generation(model_dir: str) -> None:
        from mxdiffusers.auto import detect_architecture

        try:
            arch = detect_architecture(model_dir)
        except ModelLoadError:
            return  # bare component dirs stay permissive (assumed FLUX.2/klein layout)
        if arch in _FLUX1_CLASSES:
            raise ModelLoadError(
                f"{model_dir} is a FLUX.1-generation checkpoint ({arch}). MXFluxPipeline "
                "runs FLUX.2 (klein) today; FLUX.1 schnell/dev/Kontext support is planned — "
                "see mxdiffusers/flux/FLUX1_SPEC.md."
            )

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

    def set_lora_weights(self, loras: list[tuple[str, float]]) -> dict:
        return self._engine.set_loras([(str(path), float(scale)) for path, scale in loras])

    def unload_lora_weights(self) -> None:
        self._engine.clear_loras()
