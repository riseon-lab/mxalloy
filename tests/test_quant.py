from __future__ import annotations

import numpy as np
import pytest

from mxalloy.quant import dequantize_int8_weight, quantize_int8_weight


def test_grouping_shape_divisible() -> None:
    weight = np.random.default_rng(0).standard_normal((8, 128)).astype(np.float32)
    q = quantize_int8_weight(weight, group_size=64)
    assert q.values.shape == weight.shape
    assert q.scale.shape == (8, 2)
    assert q.zero_point is None


def test_grouping_shape_with_padding() -> None:
    weight = np.random.default_rng(1).standard_normal((4, 100)).astype(np.float32)
    q = quantize_int8_weight(weight, group_size=64)
    assert q.values.shape == weight.shape
    assert q.scale.shape == (4, 2)


def test_scale_is_per_input_group() -> None:
    weight = np.concatenate(
        [np.full((1, 4), 10.0), np.full((1, 4), 0.1)], axis=1
    ).astype(np.float32)
    q = quantize_int8_weight(weight, group_size=4)
    assert q.scale.shape == (1, 2)
    assert q.scale[0, 0] > q.scale[0, 1]


def test_symmetric_round_trip_is_close() -> None:
    weight = np.random.default_rng(2).standard_normal((16, 256)).astype(np.float32)
    q = quantize_int8_weight(weight, group_size=64, symmetric=True)
    recon = dequantize_int8_weight(q)
    rel_err = np.linalg.norm(recon - weight) / np.linalg.norm(weight)
    assert rel_err < 0.02


def test_asymmetric_round_trip_is_close() -> None:
    weight = np.random.default_rng(3).standard_normal((16, 256)).astype(np.float32)
    q = quantize_int8_weight(weight, group_size=64, symmetric=False)
    assert q.zero_point is not None
    recon = dequantize_int8_weight(q)
    rel_err = np.linalg.norm(recon - weight) / np.linalg.norm(weight)
    assert rel_err < 0.02


def test_one_dimensional_weight_round_trip() -> None:
    weight = np.random.default_rng(4).standard_normal(130).astype(np.float32)
    q = quantize_int8_weight(weight, group_size=64)
    assert q.values.shape == (130,)
    assert q.scale.shape == (3,)
    recon = dequantize_int8_weight(q)
    assert np.linalg.norm(recon - weight) / np.linalg.norm(weight) < 0.03


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        quantize_int8_weight(np.array(3.0, dtype=np.float32))
    with pytest.raises(ValueError):
        quantize_int8_weight(np.zeros((4, 4), dtype=np.float32), group_size=0)
