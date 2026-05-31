"""Streaming quantized weight loader — mxalloy's core, model-agnostic memory primitive.

Streams a model's safetensors one tensor at a time into an already-built resident module,
quantizing eligible 2D weights on the fly and freeing each bf16 source immediately — so peak
memory stays near the quantized size, never the full bf16 model. This is the core of
mxalloy's memory advantage (a naive loader holds the whole bf16 model, then quantizes,
peaking at full size).

Model-specific concerns — which checkpoint keys map to which module params — are supplied by
the caller via ``remap``, so this layer is reusable across model families (diffusion, LLMs,
…). Reach it as ``mxalloy.load_quantized``.

Requires mlx; not imported by ``import mxalloy`` (kept mlx-free) — exposed lazily.
"""

from __future__ import annotations

import glob
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten


@dataclass(frozen=True, slots=True)
class QuantConfig:
    """Quantization for the streaming loader. ``bits=None`` loads bf16 (no quantization)."""

    bits: int | None = 4
    group_size: int = 64


def component_files(model_dir: str | Path, component: str) -> list[str]:
    """All safetensors shards for a sub-component dir (e.g. ``"transformer"``) of a model."""
    files = sorted(glob.glob(str(Path(model_dir) / component / "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors for component {component!r} in {model_dir}")
    return files


def _identity(key: str) -> str | None:
    return key


def load_quantized(
    module: nn.Module,
    files: list[str],
    *,
    remap: Callable[[str], str | None] = _identity,
    quant: QuantConfig | None = None,
    transpose_conv: bool = True,
) -> set[str]:
    """Stream ``files`` into resident ``module``, quantizing per ``quant``, freeing per tensor.

    ``remap(checkpoint_key) -> module_param_path | None`` maps each tensor onto a module
    param (``None`` skips it). With ``quant.bits`` set the module is ``nn.quantize``'d and
    eligible 2D weights become ``(packed, scales, biases)`` on the fly; ``bits=None`` loads
    bf16. 4D conv weights are transposed PyTorch ``[out,in,kh,kw]`` -> mlx ``[out,kh,kw,in]``
    when ``transpose_conv``. Returns module param paths left unpopulated (empty = full coverage).
    """
    quant = quant if quant is not None else QuantConfig()
    if quant.bits is not None:
        nn.quantize(
            module,
            group_size=quant.group_size,
            bits=quant.bits,
            class_predicate=lambda _path, m: hasattr(m, "to_quantized"),
        )
    targets = {k for k, _ in tree_flatten(module.parameters())}
    quantized_bases = {k[: -len(".scales")] for k in targets if k.endswith(".scales")}
    updates: list[tuple[str, mx.array]] = []
    for path in files:
        weights = mx.load(path)  # lazy: nothing materialized until visited
        for ckpt_key in list(weights.keys()):
            module_key = remap(ckpt_key)
            if module_key is None:
                weights[ckpt_key] = None
                continue
            weight = weights[ckpt_key]
            if transpose_conv and weight.ndim == 4:
                weight = weight.transpose(0, 2, 3, 1)
            base = module_key[: -len(".weight")] if module_key.endswith(".weight") else None
            if base is not None and base in quantized_bases:
                packed, scales, biases = mx.quantize(
                    weight, group_size=quant.group_size, bits=quant.bits
                )
                mx.eval(packed, scales, biases)
                updates += [
                    (f"{base}.weight", packed),
                    (f"{base}.scales", scales),
                    (f"{base}.biases", biases),
                ]
            elif module_key in targets:
                mx.eval(weight)
                updates.append((module_key, weight))
            weights[ckpt_key] = None  # release the bf16 source
        del weights
    module.update(tree_unflatten(updates))
    mx.eval(module.parameters())
    return targets - {k for k, _ in updates}
