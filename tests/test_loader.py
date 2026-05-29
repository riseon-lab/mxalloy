from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")


def test_load_quantized_streams_and_quantizes(tmp_path):
    from mxalloy.runtime.loader import StreamingQuantConfig, load_quantized

    weight = mx.random.normal((128, 256)).astype(mx.bfloat16)
    bias = mx.random.normal((128,)).astype(mx.bfloat16)
    path = tmp_path / "weights.safetensors"
    mx.save_safetensors(str(path), {"layer.weight": weight, "layer.bias": bias})

    out = load_quantized(path, StreamingQuantConfig(bits=4, group_size=64))

    # 2D weight is quantized into (packed, scales, biases).
    assert isinstance(out["layer.weight"], tuple)
    packed, scales, biases = out["layer.weight"]
    assert packed.shape[0] == 128
    assert scales.shape[0] == 128
    # 1D bias passes through unquantized.
    assert not isinstance(out["layer.bias"], tuple)
    assert out["layer.bias"].ndim == 1


def test_non_divisible_last_dim_passes_through(tmp_path):
    from mxalloy.runtime.loader import StreamingQuantConfig, load_quantized

    # last dim 100 is not divisible by group_size 64 -> not quantized
    weight = mx.random.normal((32, 100)).astype(mx.bfloat16)
    path = tmp_path / "odd.safetensors"
    mx.save_safetensors(str(path), {"odd.weight": weight})

    out = load_quantized(path, StreamingQuantConfig(bits=4, group_size=64))
    assert not isinstance(out["odd.weight"], tuple)
    assert out["odd.weight"].shape == (32, 100)
