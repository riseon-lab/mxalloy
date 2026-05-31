"""Fused quantized-KV scaled-dot-product attention (Python surface + reference).

Phase-2 capability: attention where K and V live in MLX's affine int4/int8 *group* format
(``mx.quantize`` -> packed uint32 weights + per-group ``scales`` + ``biases``) and are
dequantized **inline** during the attention pass, so the 16-bit K/V never materialise on
the unified-memory heap. The target is long-context / KV-cached / batched-serving
workloads where that transient dequant spike dominates.

Scope note (measured): this does **not** speed up the current klein 4-step txt2img path.
That path has no KV cache (the diffusion transformer recomputes Q/K/V every step) and is
GEMM-bound -- attention is ~0.7% of a step. The fused kernel pays off where attention is a
real memory/latency cost: autoregressive decode, long context, and continuous batching.

Two backends behind one API:

* ``_fallback`` -- pure MLX: ``mx.dequantize`` K/V, then
  ``mx.fast.scaled_dot_product_attention``. Always available (no Metal toolchain needed),
  numerically correct, and the **oracle** the fused kernel is verified against. It *does*
  pay the transient dequant spike -- that spike is exactly what the kernel removes.
* compiled ``mxalloy.attention._ext`` -- the fused metallib kernel. EXPERIMENTAL: the
  sources live under ``research/attention_kernel/`` (frozen) and are not built or shipped by
  default. It is numerically correct but, on the current GEMM-bound diffusion path, it is
  memory-not-speed -- so the pure-MLX fallback above is the live primitive. Build it only to
  A/B KV-cached / long-context workloads, dropping the artifact next to this module.

INTERNAL: requires mlx. The pure-MLX fallback needs no Metal toolchain.
"""

from __future__ import annotations

from typing import NamedTuple

import mlx.core as mx


class QuantizedKV(NamedTuple):
    """A K or V tensor in MLX affine-quant group format, as returned by ``mx.quantize``.

    Quantization is along the last (head_dim) axis. ``weights`` is bit-packed into uint32.
    """

    weights: mx.array  # packed uint32, shape (..., head_dim * bits / 32)
    scales: mx.array  # per-group, shape (..., head_dim / group_size)
    biases: mx.array
    group_size: int = 64
    bits: int = 8  # 8-bit KV is ~lossless (<1%, >40dB); 4-bit (~12%) is the OOM-only lever


# The compiled fused kernel is optional and off by default: absent unless built (from
# research/attention_kernel, needs the Metal toolchain) and dropped next to this module.
# Degrade to the pure-MLX fallback when it isn't there -- the shipped default. The hasattr
# guard rejects a stray namespace package and only accepts a real compiled module.
try:
    from mxalloy.attention import _ext as _compiled  # type: ignore

    if not hasattr(_compiled, "quantized_scaled_dot_product_attention"):
        _compiled = None
except Exception:  # noqa: BLE001
    _compiled = None


def quantize_kv(x: mx.array, group_size: int = 64, bits: int = 8) -> QuantizedKV:
    """Quantize a dense (..., head_dim) K/V tensor into the group format the kernel reads."""
    weights, scales, biases = mx.quantize(x, group_size=group_size, bits=bits)
    return QuantizedKV(weights, scales, biases, group_size, bits)


def _dequant(t: QuantizedKV) -> mx.array:
    return mx.dequantize(t.weights, t.scales, t.biases, group_size=t.group_size, bits=t.bits)


def _fallback(q: mx.array, k: QuantizedKV, v: QuantizedKV, *, scale: float, mask) -> mx.array:
    # Materialises dequantized K/V (the spike the kernel avoids), then the fused MLX SDPA.
    return mx.fast.scaled_dot_product_attention(
        q, _dequant(k), _dequant(v), scale=scale, mask=mask
    )


def kernel_available(q_dtype: mx.Dtype, head_dim: int, k: QuantizedKV, has_mask: bool) -> bool:
    """Whether the compiled fused kernel is built *and* supports this configuration."""
    return _compiled is not None and _compiled.supports(
        q_dtype, head_dim, k.group_size, k.bits, has_mask
    )


def quantized_scaled_dot_product_attention(
    q: mx.array,
    k: QuantizedKV,
    v: QuantizedKV,
    *,
    scale: float,
    mask=None,
    prefer_kernel: bool = True,
) -> mx.array:
    """``O = softmax(Q Kᵀ · scale) V`` with K, V supplied in quantized group format.

    Args:
        q: dense queries ``(B, H, L, D)`` in fp16/bf16.
        k, v: :class:`QuantizedKV` over the same logical ``(B, H, S, D)``, quantized along D.
        scale: softmax scale (typically ``1/sqrt(D)``).
        mask: optional ``None | "causal" | array``.
        prefer_kernel: use the fused kernel when built + applicable; else the fallback.

    Returns:
        ``(B, H, L, D)`` dense output.
    """
    if prefer_kernel and kernel_available(q.dtype, q.shape[-1], k, mask is not None):
        return _compiled.quantized_scaled_dot_product_attention(
            q,
            k.weights,
            k.scales,
            k.biases,
            v.weights,
            v.scales,
            v.biases,
            scale=scale,
            group_size=k.group_size,
            bits=k.bits,
            mask=mask,
        )
    return _fallback(q, k, v, scale=scale, mask=mask)
