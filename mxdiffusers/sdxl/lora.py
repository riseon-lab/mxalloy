"""SDXL LoRA key mapping over the shared ``mxdiffusers.lora`` core.

Handles diffusers/PEFT-format SDXL LoRAs (``unet.<path>.lora_A/.lora_B``). Because the SDXL
module tree mirrors the diffusers state_dict exactly, mapping is prefix-stripping plus a
Linear-target whitelist. Text-encoder LoRA keys and kohya-flattened (``lora_unet_…``
underscore) names are documented TODOs.

INTERNAL: requires mlx (except :func:`target_paths_for_lora_base`, which is pure).
"""

from __future__ import annotations

from typing import Any

from mlx import nn

from mxdiffusers.lora import LoRALinear, clear_loras, load_lora_file
from mxdiffusers.lora import apply_loras as _apply_loras

__all__ = [
    "LoRALinear",
    "apply_loras",
    "clear_loras",
    "load_lora_file",
    "target_paths_for_lora_base",
]

_PREFIXES = ("base_model.model.", "unet.", "model.")

_LINEAR_SUFFIXES = (
    ".attn1.to_q",
    ".attn1.to_k",
    ".attn1.to_v",
    ".attn1.to_out.0",
    ".attn2.to_q",
    ".attn2.to_k",
    ".attn2.to_v",
    ".attn2.to_out.0",
    ".ff.net.0.proj",
    ".ff.net.2",
    ".proj_in",
    ".proj_out",
    ".time_emb_proj",
)


def _strip_prefixes(path: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if path.startswith(prefix):
                path = path[len(prefix) :]
                changed = True
    return path


def target_paths_for_lora_base(base: str) -> list[str]:
    """Map a LoRA base key to native SDXL UNet Linear module paths (pure, testable).

    Text-encoder LoRA keys (``text_encoder.``/``te1_``/…) return [] for now. The module tree
    mirrors diffusers, so a mapped path is the stripped path itself.
    """
    if base.startswith(("text_encoder", "te1", "te2", "lora_te")):
        return []
    path = _strip_prefixes(base)
    if path.endswith(".to_out"):
        path = f"{path}.0"
    return [path] if path.endswith(_LINEAR_SUFFIXES) else []


def _targets_for_base(base: str) -> tuple[list[tuple[str, int | None]], str]:
    return [(p, None) for p in target_paths_for_lora_base(base)], _strip_prefixes(base)


def apply_loras(unet: nn.Module, loras: list[tuple[dict[str, Any], float]]) -> dict:
    """Replace active LoRA deltas on the resident SDXL UNet (hot-swap)."""
    return _apply_loras(unet, loras, targets_for_base=_targets_for_base)
