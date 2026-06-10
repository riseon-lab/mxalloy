"""Z-Image LoRA key mapping over the shared ``mxdiffusers.lora`` core.

Accepts common diffusers/PEFT-style safetensors keys and maps them onto the native
``ZImageTransformer`` module paths. LoRA tensors stay separate from the quantized base
weights, so adapters hot-swap without reloading the 6B model.

INTERNAL: requires mlx (except :func:`target_paths_for_lora_base`, which is pure).
"""

from __future__ import annotations

from typing import Any

from mlx import nn

from mxdiffusers.lora import LoRALinear, clear_loras, load_lora_file
from mxdiffusers.lora import apply_loras as _apply_loras
from mxdiffusers.zimage.weight_mapping import remap_zimage_transformer_key

__all__ = [
    "LoRALinear",
    "apply_loras",
    "clear_loras",
    "load_lora_file",
    "target_paths_for_lora_base",
]

_PREFIXES = (
    "base_model.model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "transformer.",
    "model.",
)

_TOP_LEVEL_LINEAR_TARGETS = {
    "x_embedder",
    "cap_embedder.proj",
    "t_embedder.l1",
    "t_embedder.l2",
    "final_layer.linear",
    "final_layer.adaLN_proj",
}

_BLOCK_LINEAR_SUFFIXES = (
    ".attention.to_q",
    ".attention.to_k",
    ".attention.to_v",
    ".attention.to_out.0",
    ".feed_forward.w1",
    ".feed_forward.w2",
    ".feed_forward.w3",
    ".adaLN_modulation.0",
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


def _normalize_target_path(path: str) -> str:
    if path.endswith(".attention.to_out"):
        return f"{path}.0"
    return path


def _is_linear_target(path: str) -> bool:
    return path in _TOP_LEVEL_LINEAR_TARGETS or path.endswith(_BLOCK_LINEAR_SUFFIXES)


def target_paths_for_lora_base(base: str) -> list[str]:
    """Map a LoRA base key to native Z-Image Linear module paths.

    Returns an empty list for unsupported conventions or non-Linear targets. Kept pure so
    key-mapping coverage can be tested without loading mlx weights.
    """
    path = _strip_prefixes(base)
    mapped = remap_zimage_transformer_key(f"{path}.weight")
    if mapped is not None and mapped.endswith(".weight"):
        path = mapped[: -len(".weight")]
    path = _normalize_target_path(path)
    return [path] if _is_linear_target(path) else []


def _targets_for_base(base: str) -> tuple[list[tuple[str, int | None]], str]:
    return [(p, None) for p in target_paths_for_lora_base(base)], _strip_prefixes(base)


def apply_loras(transformer: nn.Module, loras: list[tuple[dict[str, Any], float]]) -> dict:
    """Replace active LoRA deltas on the resident Z-Image transformer (hot-swap)."""
    return _apply_loras(transformer, loras, targets_for_base=_targets_for_base)
