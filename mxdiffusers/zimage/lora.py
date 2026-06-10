"""Runtime LoRA support for Z-Image's MLX transformer.

The loader accepts common diffusers/PEFT-style safetensors keys and maps them onto the
native ``ZImageTransformer`` module paths. LoRA tensors stay separate from the quantized base
weights, so adapters can be hot-swapped without reloading the 6B model.

INTERNAL: requires mlx.
"""

from __future__ import annotations

import re
from typing import Any

import mlx.core as mx
from mlx import nn

from mxdiffusers.zimage.weight_mapping import remap_zimage_transformer_key

PRECISION = mx.bfloat16

_LORA_WEIGHT_RE = re.compile(
    r"^(?P<base>.+)\."
    r"(?P<kind>lora_A|lora_B|lora_down|lora_up|lora\.down|lora\.up)"
    r"(?:\.[^.]+)?\.weight$"
)
_LORA_ALPHA_RE = re.compile(r"^(?P<base>.+)\.(?:alpha|lora_alpha|network_alpha)$")

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


class LoRALinear(nn.Module):
    """Wrap a Linear-like module and add one or more low-rank deltas at call time."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        object.__setattr__(self, "_loras", [])

    def add(self, a: mx.array, b: mx.array, scale: float) -> None:
        self._loras.append((a.astype(PRECISION), b.astype(PRECISION), float(scale)))

    def clear(self) -> None:
        self._loras.clear()

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        for a, b, scale in self._loras:
            y = y + scale * ((x @ a.T) @ b.T)
        return y


def _get(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _set(root: Any, path: str, val: Any) -> None:
    parts = path.split(".")
    obj = root
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = val
    else:
        setattr(obj, last, val)


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


def _wrap(transformer: nn.Module, path: str) -> LoRALinear:
    reg = getattr(transformer, "_lora_wrappers", None)
    if reg is None:
        reg = {}
        object.__setattr__(transformer, "_lora_wrappers", reg)
    if path in reg:
        return reg[path]
    base = _get(transformer, path)
    wrapper = base if isinstance(base, LoRALinear) else LoRALinear(base)
    if not isinstance(base, LoRALinear):
        _set(transformer, path, wrapper)
    reg[path] = wrapper
    return wrapper


def clear_loras(transformer: nn.Module) -> None:
    for wrapper in getattr(transformer, "_lora_wrappers", {}).values():
        wrapper.clear()


def _group(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, value in state.items():
        if match := _LORA_WEIGHT_RE.match(key):
            kind = match.group("kind")
            slot = "A" if kind in {"lora_A", "lora_down", "lora.down"} else "B"
            out.setdefault(match.group("base"), {})[slot] = value
        elif match := _LORA_ALPHA_RE.match(key):
            out.setdefault(match.group("base"), {})["alpha"] = float(value)
    return out


def apply_loras(transformer: nn.Module, loras: list[tuple[dict[str, Any], float]]) -> dict:
    """Replace active LoRA deltas on the resident Z-Image transformer.

    ``loras`` is ``[(state_dict, strength), ...]``. Tensor dtype can be fp32, fp16, or bf16;
    adapters are cast to bf16 inside the wrapper before use.
    """
    clear_loras(transformer)
    applied = 0
    skipped: set[str] = set()
    for state, strength in loras:
        for base, ab in _group(state).items():
            if "A" not in ab or "B" not in ab:
                continue
            a, b = ab["A"], ab["B"]
            if a.ndim != 2 or b.ndim != 2:
                skipped.add(_strip_prefixes(base))
                continue
            rank = max(1, int(a.shape[0]))
            alpha = ab.get("alpha")
            scale = float(strength) * (float(alpha) / rank if alpha is not None else 1.0)
            targets = target_paths_for_lora_base(base)
            if not targets:
                skipped.add(_strip_prefixes(base))
                continue
            for path in targets:
                try:
                    _wrap(transformer, path).add(a, b, scale)
                    applied += 1
                except (AttributeError, IndexError, TypeError):
                    skipped.add(path)
    return {"applied": applied, "skipped": sorted(skipped)}


def load_lora_file(path: str) -> dict[str, Any]:
    """Load a LoRA safetensors/MLX-supported checkpoint into a name->array dict."""
    return mx.load(path)
