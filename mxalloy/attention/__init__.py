"""Attention execution strategies.

Houses mxalloy's attention backends. The headline one is fused quantized-KV attention
(:func:`quantized_scaled_dot_product_attention`) -- inline-dequant int4/int8 K/V so the
16-bit tensors never hit the heap. Exposed lazily so importing this package stays cheap and
the top-level ``import mxalloy`` remains mlx-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["QuantizedKV", "quantize_kv", "quantized_scaled_dot_product_attention"]

if TYPE_CHECKING:
    from mxalloy.attention.quantized_sdpa import (
        QuantizedKV,
        quantize_kv,
        quantized_scaled_dot_product_attention,
    )


def __getattr__(name: str):  # PEP 562: lazy, mlx-only-on-use
    if name in __all__:
        from mxalloy.attention import quantized_sdpa

        return getattr(quantized_sdpa, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
