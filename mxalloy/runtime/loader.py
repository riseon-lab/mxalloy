"""Streaming quantized weight loader.

Loads safetensors weights one tensor at a time, quantizing eligible weights immediately
and freeing the bf16 source, so peak memory stays near the quantized size instead of the
full bf16 model. This is the core of mxalloy's memory advantage: a naive loader (e.g.
mflux) loads the full bf16 model and only then quantizes, peaking at the full size; this
loader never holds more than one transient bf16 tensor beyond the accumulated quantized
weights.

Relies on ``mx.load`` being lazy — tensors are not materialized until evaluated, so
unvisited weights cost nothing.

Requires mlx. Import this module only where mlx is installed; it is intentionally not
re-exported from ``mxalloy`` or ``mxalloy.runtime`` so that ``import mxalloy`` stays
free of the mlx dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

# (packed weights, scales, biases) as returned by mx.quantize.
QuantizedWeight = tuple[mx.array, mx.array, mx.array]


@dataclass(frozen=True, slots=True)
class StreamingQuantConfig:
    bits: int = 4
    group_size: int = 64


def _quantizable(weight: mx.array, group_size: int) -> bool:
    """A 2D weight whose contraction (last) axis divides evenly into groups."""
    return weight.ndim == 2 and weight.shape[-1] % group_size == 0


def load_quantized(
    path: str | Path,
    config: StreamingQuantConfig | None = None,
) -> dict[str, mx.array | QuantizedWeight]:
    """Stream a safetensors file into quantized MLX weights, freeing bf16 per tensor.

    Eligible 2D weights become ``(packed, scales, biases)`` tuples; everything else
    (norms, biases, odd shapes) passes through unquantized.
    """
    config = config or StreamingQuantConfig()
    weights = mx.load(str(path))  # lazy: nothing materialized yet
    out: dict[str, mx.array | QuantizedWeight] = {}
    for name in list(weights.keys()):
        weight = weights[name]
        if _quantizable(weight, config.group_size):
            packed, scales, biases = mx.quantize(
                weight, group_size=config.group_size, bits=config.bits
            )
            mx.eval(packed, scales, biases)
            out[name] = (packed, scales, biases)
        else:
            mx.eval(weight)
            out[name] = weight
        weights[name] = None  # release the bf16 source so it can be freed
        del weight
    return out
