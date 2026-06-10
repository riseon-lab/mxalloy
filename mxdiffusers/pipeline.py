"""MXPipeline — the diffusers-style base every mxdiffusers model family builds on.

A pipeline owns the resident model components and the denoise loop, and delegates device
detection, memory planning, and quantized loading to the **mxalloy** runtime. Concrete
families (``MXFluxPipeline``, ``MXZimagePipeline``, …) subclass this and supply their model
graph + scheduler + latent handling.

The surface intentionally mirrors 🤗 diffusers — ``from_pretrained`` + ``__call__`` returning a
result whose ``.images`` is a list — so it reads like the framework people already know, while
the optimization underneath is mxalloy. INTERNAL until the API stabilises; requires mlx on use.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image

# (completed_step_1based, total_steps) — lets a caller (e.g. the tester UI) show progress
# without reaching into a family's denoise loop.
OnStep = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class MXResult:
    """Generation result, diffusers-style: ``result.images[0]`` (or ``result.image``)."""

    images: list[Image.Image]
    seed: int

    @property
    def image(self) -> Image.Image:
        return self.images[0]


class MXPipeline:
    """Base diffusion pipeline. Subclass per model family (see ``MXFluxPipeline``)."""

    family: str = "base"

    @classmethod
    def from_pretrained(cls, model_id: str | None = None, **kwargs) -> MXPipeline:
        """Load + quantize the model's components via mxalloy and keep them resident."""
        raise NotImplementedError

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
        """Run text-to-image: encode → denoise loop → decode. Returns an :class:`MXResult`."""
        raise NotImplementedError

    # diffusers-familiar LoRA surface; families that support it override these.
    def load_lora_weights(self, path: str, *, scale: float = 1.0) -> dict:
        raise NotImplementedError(f"{type(self).__name__} does not support LoRA")

    def set_lora_weights(self, loras: list[tuple[str, float]]) -> dict:
        if len(loras) == 1:
            path, scale = loras[0]
            return self.load_lora_weights(path, scale=scale)
        raise NotImplementedError(f"{type(self).__name__} does not support multi-LoRA")

    def unload_lora_weights(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support LoRA")

    @staticmethod
    def device():
        """The detected Apple Silicon device profile (from the mxalloy runtime)."""
        from mxalloy.runtime import detect_device

        return detect_device()
