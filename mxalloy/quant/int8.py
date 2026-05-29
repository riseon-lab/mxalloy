"""Initial INT8 quantization primitives.

Weights are grouped along the last (input/contraction) axis so that scales line up
with a matmul's reduction dimension. For a Linear weight of shape ``(out, in)`` this
means each output row is split into ``ceil(in / group_size)`` groups, each with its own
scale. Dequantization then fuses cleanly into ``y = W @ x``.

The implementation is framework-neutral (numpy) so calibration and tests can land before
MLX-specific packing and fused dequantization are added.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mxalloy.errors import QuantizationError


@dataclass(frozen=True, slots=True)
class Int8QuantizedWeight:
    """Grouped INT8 weight.

    ``values`` keeps the original weight shape. ``scale`` (and ``zero_point`` for the
    asymmetric path) carry one entry per group along the last axis, with shape
    ``(*weight.shape[:-1], num_groups)`` where ``num_groups = ceil(in / group_size)``.
    """

    values: np.ndarray
    scale: np.ndarray
    zero_point: np.ndarray | None = None
    group_size: int = 64


def quantize_int8_weight(
    weight: np.ndarray,
    *,
    group_size: int = 64,
    symmetric: bool = True,
) -> Int8QuantizedWeight:
    """Quantize a weight tensor into grouped INT8 values along the input axis."""
    if weight.ndim == 0:
        raise QuantizationError(
            f"Cannot quantize a 0-d weight; expected at least 1 dimension, "
            f"got shape {weight.shape}."
        )
    if group_size <= 0:
        raise QuantizationError(f"group_size must be a positive integer; got {group_size}.")

    w = weight.astype(np.float32, copy=False)
    leading = w.shape[:-1]
    in_features = w.shape[-1]
    pad = (-in_features) % group_size
    if pad:
        pad_width = [(0, 0)] * (w.ndim - 1) + [(0, pad)]
        w = np.pad(w, pad_width, mode="constant")
    num_groups = w.shape[-1] // group_size

    grouped = w.reshape(*leading, num_groups, group_size)

    if symmetric:
        max_abs = np.maximum(np.max(np.abs(grouped), axis=-1, keepdims=True), 1e-8)
        scale = max_abs / 127.0
        values = np.clip(np.round(grouped / scale), -127, 127).astype(np.int8)
        return Int8QuantizedWeight(
            values=values.reshape(*leading, num_groups * group_size)[..., :in_features],
            scale=scale.reshape(*leading, num_groups),
            group_size=group_size,
        )

    min_value = np.min(grouped, axis=-1, keepdims=True)
    max_value = np.max(grouped, axis=-1, keepdims=True)
    scale = np.maximum((max_value - min_value) / 255.0, 1e-8)
    zero_point = np.clip(np.round(-min_value / scale), 0, 255)
    values = np.clip(np.round(grouped / scale + zero_point), 0, 255).astype(np.uint8)
    return Int8QuantizedWeight(
        values=values.reshape(*leading, num_groups * group_size)[..., :in_features],
        scale=scale.reshape(*leading, num_groups),
        zero_point=zero_point.reshape(*leading, num_groups),
        group_size=group_size,
    )


def dequantize_int8_weight(quantized: Int8QuantizedWeight) -> np.ndarray:
    """Reconstruct an FP32 weight from a grouped INT8 representation."""
    in_features = quantized.values.shape[-1]
    scale = np.repeat(quantized.scale, quantized.group_size, axis=-1)[..., :in_features]
    values = quantized.values.astype(np.float32)
    if quantized.zero_point is None:
        return values * scale
    zero_point = np.repeat(quantized.zero_point, quantized.group_size, axis=-1)[..., :in_features]
    return (values - zero_point) * scale
