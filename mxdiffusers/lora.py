"""Shared runtime-LoRA core: hot-swap low-rank deltas on a quantized resident model.

A LoRA-adapted Linear computes ``y = base(x) + Σ scaleᵢ · (x @ Aᵢᵀ) @ Bᵢᵀ`` with the base
weights left quantized — adapters apply at runtime and swap without reloading the model.

This module is convention-agnostic. Each model family supplies a ``targets_for_base``
callable that maps a LoRA state-dict base key onto its native module paths (and, for
fused-QKV conventions, which third of ``B`` to slice). Grouping handles the common key
styles in one place: PEFT/diffusers (``.lora_A/.lora_B``), kohya-suffix
(``.lora_down/.lora_up``), and ``alpha``/``lora_alpha``/``network_alpha``.

INTERNAL: requires mlx.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import mlx.core as mx
from mlx import nn

PRECISION = mx.bfloat16

# (base, kind) for "<base>.lora_A[.<adapter>].weight" and the lora_down/up spellings.
_LORA_WEIGHT_RE = re.compile(
    r"^(?P<base>.+)\."
    r"(?P<kind>lora_A|lora_B|lora_down|lora_up|lora\.down|lora\.up)"
    r"(?:\.[^.]+)?\.weight$"
)
_LORA_ALPHA_RE = re.compile(r"^(?P<base>.+)\.(?:alpha|lora_alpha|network_alpha)$")

# targets_for_base(base_key) -> ([(module_path, qkv_third | None), ...], display_name)
# qkv_third in {0,1,2} slices rows [third*out/3:(third+1)*out/3] of B (fused QKV sharing A).
TargetsForBase = Callable[[str], tuple[list[tuple[str, int | None]], str]]


class LoRALinear(nn.Module):
    """Wraps a (quantized) Linear, adding zero or more low-rank deltas at call time."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        # Held off the parameter tree (never trained/optimised): plain attribute.
        object.__setattr__(self, "_loras", [])  # list of (A (r,in), B (out,r), scale)

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


def wrap_lora(module: nn.Module, path: str) -> LoRALinear:
    """Wrap the Linear at ``path`` in a LoRALinear (idempotent, registry-cached)."""
    reg = getattr(module, "_lora_wrappers", None)
    if reg is None:
        reg = {}
        object.__setattr__(module, "_lora_wrappers", reg)
    if path in reg:
        return reg[path]
    base = _get(module, path)
    wrapper = base if isinstance(base, LoRALinear) else LoRALinear(base)
    if not isinstance(base, LoRALinear):
        _set(module, path, wrapper)
    reg[path] = wrapper
    return wrapper


def clear_loras(module: nn.Module) -> None:
    """Remove all active LoRA deltas (the wrapped base weights are untouched)."""
    for wrapper in getattr(module, "_lora_wrappers", {}).values():
        wrapper.clear()


def group_lora_state(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """state_dict -> {base_key: {'A': arr, 'B': arr, 'alpha': float | None}}."""
    out: dict[str, dict[str, Any]] = {}
    for key, value in state.items():
        if match := _LORA_WEIGHT_RE.match(key):
            kind = match.group("kind")
            slot = "A" if kind in {"lora_A", "lora_down", "lora.down"} else "B"
            out.setdefault(match.group("base"), {})[slot] = value
        elif match := _LORA_ALPHA_RE.match(key):
            out.setdefault(match.group("base"), {})["alpha"] = float(value)
    return out


def apply_loras(
    module: nn.Module,
    loras: list[tuple[dict[str, Any], float]],
    *,
    targets_for_base: TargetsForBase,
) -> dict:
    """Replace all active LoRA deltas on ``module`` with ``loras``.

    ``loras`` is ``[(state_dict, strength), ...]`` — each call replaces the full active set
    (hot-swap: the quantized base is untouched). Tensors may be fp32/fp16/bf16; adapters are
    cast to bf16 in the wrapper. Returns ``{'applied': n_layers, 'skipped': [names]}``.
    """
    clear_loras(module)
    applied = 0
    skipped: set[str] = set()
    for state, strength in loras:
        for base, ab in group_lora_state(state).items():
            if "A" not in ab or "B" not in ab:
                continue
            a, b = ab["A"], ab["B"]
            targets, display = targets_for_base(base)
            if a.ndim != 2 or b.ndim != 2:
                skipped.add(display)
                continue
            rank = max(1, int(a.shape[0]))
            alpha = ab.get("alpha")
            scale = float(strength) * (float(alpha) / rank if alpha is not None else 1.0)
            if not targets:
                skipped.add(display)
                continue
            for path, third in targets:
                bt = (
                    b
                    if third is None
                    else b[third * (b.shape[0] // 3) : (third + 1) * (b.shape[0] // 3)]
                )
                try:
                    wrap_lora(module, path).add(a, bt, scale)
                    applied += 1
                except (AttributeError, IndexError, TypeError):
                    skipped.add(path)
    return {"applied": applied, "skipped": sorted(skipped)}


def load_lora_file(path: str) -> dict[str, Any]:
    """Load a LoRA safetensors/MLX-supported checkpoint into a name->array dict."""
    return mx.load(path)
