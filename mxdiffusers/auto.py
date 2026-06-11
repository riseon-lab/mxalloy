"""MXAutoPipeline — route a checkpoint to its architecture pipeline.

Detection reads the checkpoint itself (offline, mlx-free): ``model_index.json``'s
``_class_name`` when present (canonical diffusers layout), falling back to the denoiser's
``config.json`` for component-only snapshots. Unknown-but-recognized architectures raise
``ModelLoadError`` naming the architecture and its status, so "not supported yet" is a clear
message instead of a shape error.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from mxalloy.errors import ModelLoadError
from mxdiffusers.hub import resolve_model_dir

# diffusers pipeline _class_name -> the MX architecture-family pipeline (module:Class).
# The FLUX family class owns generation handling itself: FLUX.2 runs, FLUX.1 reports its
# planned status — so all Flux* names route to it.
_PIPELINES: dict[str, str] = {
    "StableDiffusionXLPipeline": "mxdiffusers.sdxl.pipeline:MXSDXLPipeline",
    "ZImagePipeline": "mxdiffusers.zimage.pipeline:MXZimagePipeline",
    "Flux2Pipeline": "mxdiffusers.flux.pipeline:MXFluxPipeline",
    "FluxPipeline": "mxdiffusers.flux.pipeline:MXFluxPipeline",
    "FluxKontextPipeline": "mxdiffusers.flux.pipeline:MXFluxPipeline",
}
# denoiser config _class_name -> implemented MX pipeline (component-only snapshots)
_DENOISERS: dict[str, str] = {
    "UNet2DConditionModel": "mxdiffusers.sdxl.pipeline:MXSDXLPipeline",  # checked for SDXL below
    "ZImageTransformer2DModel": "mxdiffusers.zimage.pipeline:MXZimagePipeline",
    "Flux2Transformer2DModel": "mxdiffusers.flux.pipeline:MXFluxPipeline",
}
# recognized but not implemented -> honest status in the error message
_PLANNED: dict[str, str] = {
    "StableDiffusion3Pipeline": "SD3 / SD3.5 — planned; see mxdiffusers/sd3/SPEC.md",
    "QwenImagePipeline": "Qwen-Image — planned; see mxdiffusers/qwen_image/SPEC.md",
    "StableDiffusionPipeline": "SD 1.5/2.x — not planned for v1",
}


def detect_architecture(model_dir: str) -> str:
    """Return the diffusers pipeline class name this checkpoint declares."""
    index = Path(model_dir) / "model_index.json"
    if index.is_file():
        name = json.loads(index.read_text()).get("_class_name")
        if isinstance(name, str) and name:
            return name
    for component in ("transformer", "unet"):
        config = Path(model_dir) / component / "config.json"
        if not config.is_file():
            continue
        cls = json.loads(config.read_text()).get("_class_name", "")
        if cls == "UNet2DConditionModel":
            # SDXL vs SD1.5/2.x share the class; SDXL is the text_time variant.
            cfg = json.loads(config.read_text())
            if cfg.get("addition_embed_type") == "text_time":
                return "StableDiffusionXLPipeline"
            return "StableDiffusionPipeline"
        if cls in _DENOISERS:
            target = _DENOISERS[cls]
            for pipeline_name, t in _PIPELINES.items():
                if t == target:
                    return pipeline_name
    raise ModelLoadError(
        f"could not detect a diffusion architecture in {model_dir} "
        "(no model_index.json _class_name, no recognizable transformer/unet config)"
    )


class MXAutoPipeline:
    """``MXAutoPipeline.from_pretrained(id_or_dir)`` -> the right architecture pipeline."""

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: Any):
        model_dir = resolve_model_dir(model_id, default_repo=model_id)
        arch = detect_architecture(model_dir)
        target = _PIPELINES.get(arch)
        if target is None:
            status = _PLANNED.get(arch, "unknown architecture")
            supported = ", ".join(sorted(_PIPELINES))
            raise ModelLoadError(
                f"{arch} checkpoints are not supported yet ({status}). "
                f"Implemented architectures: {supported}."
            )
        module_path, class_name = target.split(":")
        pipeline_cls = getattr(importlib.import_module(module_path), class_name)
        return pipeline_cls.from_pretrained(model_dir, **kwargs)
