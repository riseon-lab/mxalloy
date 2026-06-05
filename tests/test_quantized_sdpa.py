"""Fused quantized-KV attention -- Python surface + pure-MLX fallback.

These exercise the path that works today (no Metal toolchain): the QuantizedKV format and
the dequant->SDPA fallback. When the compiled kernel lands, it's verified against this same
fallback as the oracle.
"""

from __future__ import annotations

from tests._mlx import require_mlx_core

mx = require_mlx_core()
from mxalloy.attention import (
    QuantizedKV,
    quantize_kv,
    quantized_scaled_dot_product_attention,
)
from mxalloy.attention.quantized_sdpa import _dequant


def test_quantize_kv_roundtrip_shapes() -> None:
    x = mx.random.normal((1, 4, 128, 128)).astype(mx.float16)
    kv = quantize_kv(x, group_size=64, bits=4)
    assert isinstance(kv, QuantizedKV)
    assert (kv.group_size, kv.bits) == (64, 4)
    deq = _dequant(kv)
    assert deq.shape == x.shape and deq.dtype == x.dtype


def test_fallback_matches_explicit_dequant_then_sdpa() -> None:
    mx.random.seed(0)
    b, h, length, d = 1, 8, 64, 128
    q = mx.random.normal((b, h, length, d)).astype(mx.float16)
    k = mx.random.normal((b, h, length, d)).astype(mx.float16)
    v = mx.random.normal((b, h, length, d)).astype(mx.float16)
    scale = 1.0 / (d**0.5)
    kq, vq = quantize_kv(k), quantize_kv(v)
    out = quantized_scaled_dot_product_attention(q, kq, vq, scale=scale)  # _ext absent -> fallback
    ref = mx.fast.scaled_dot_product_attention(q, _dequant(kq), _dequant(vq), scale=scale)
    assert mx.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_8bit_kv_close_to_full_precision() -> None:
    mx.random.seed(1)
    b, h, length, d = 1, 4, 128, 128
    q = mx.random.normal((b, h, length, d)).astype(mx.float16)
    k = mx.random.normal((b, h, length, d)).astype(mx.float16)
    v = mx.random.normal((b, h, length, d)).astype(mx.float16)
    scale = 1.0 / (d**0.5)
    full = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    quant = quantized_scaled_dot_product_attention(
        q, quantize_kv(k, bits=8), quantize_kv(v, bits=8), scale=scale
    )
    assert float(mx.max(mx.abs(full - quant))) < 0.05


def test_mask_path_runs_and_keeps_shape() -> None:
    b, h, length, d = 1, 2, 32, 128
    q = mx.random.normal((b, h, length, d)).astype(mx.float16)
    k = mx.random.normal((b, h, length, d)).astype(mx.float16)
    v = mx.random.normal((b, h, length, d)).astype(mx.float16)
    mask = mx.zeros((length, length)).astype(mx.float16)
    out = quantized_scaled_dot_product_attention(
        q, quantize_kv(k), quantize_kv(v), scale=1.0 / (d**0.5), mask=mask
    )
    assert out.shape == (b, h, length, d)
