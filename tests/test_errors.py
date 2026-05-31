from __future__ import annotations

import pytest

from mxalloy.errors import (
    AlloyError,
    ConfigurationError,
    IncompatibleLoRAError,
    ModelLoadError,
    QuantizationError,
    UnsupportedHardwareError,
)


def test_specific_errors_derive_from_alloy_error() -> None:
    for exc in (
        ConfigurationError,
        QuantizationError,
        IncompatibleLoRAError,
        UnsupportedHardwareError,
        ModelLoadError,
    ):
        assert issubclass(exc, AlloyError)


def test_value_like_errors_are_also_value_errors() -> None:
    for exc in (ConfigurationError, QuantizationError, IncompatibleLoRAError):
        assert issubclass(exc, ValueError)


def test_runtime_like_errors_are_also_runtime_errors() -> None:
    for exc in (UnsupportedHardwareError, ModelLoadError):
        assert issubclass(exc, RuntimeError)


def test_errors_are_catchable_via_base_and_builtin() -> None:
    with pytest.raises(AlloyError):
        raise QuantizationError("nope")
    with pytest.raises(ValueError):
        raise ConfigurationError("nope")
    with pytest.raises(RuntimeError):
        raise ModelLoadError("nope")
