from __future__ import annotations

import numpy as np
import pytest

from mxalloy.errors import AlloyError, ConfigurationError, QuantizationError
from mxalloy.quant import quantize_int8_weight


def test_specific_errors_derive_from_alloy_error() -> None:
    assert issubclass(QuantizationError, AlloyError)
    assert issubclass(ConfigurationError, AlloyError)


def test_value_like_errors_are_also_value_errors() -> None:
    assert issubclass(QuantizationError, ValueError)
    assert issubclass(ConfigurationError, ValueError)


def test_quantize_raises_quantization_error_catchable_three_ways() -> None:
    bad = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(QuantizationError):
        quantize_int8_weight(bad, group_size=0)
    with pytest.raises(AlloyError):
        quantize_int8_weight(bad, group_size=0)
    with pytest.raises(ValueError):
        quantize_int8_weight(bad, group_size=0)
