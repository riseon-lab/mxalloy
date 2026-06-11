from __future__ import annotations

import pytest

from mxalloy.errors import AlloyError, ConfigurationError, ModelLoadError


def test_specific_errors_derive_from_alloy_error() -> None:
    for exc in (ConfigurationError, ModelLoadError):
        assert issubclass(exc, AlloyError)


def test_errors_are_catchable_via_base_and_builtin() -> None:
    # ConfigurationError is also a ValueError; ModelLoadError is also a RuntimeError — so
    # existing builtin except-clauses keep working alongside `except AlloyError`.
    with pytest.raises(AlloyError):
        raise ConfigurationError("nope")
    with pytest.raises(ValueError):
        raise ConfigurationError("nope")
    with pytest.raises(RuntimeError):
        raise ModelLoadError("nope")


def test_configuration_error_is_raised_by_the_runtime() -> None:
    # The hierarchy is wired, not decorative: bad planner/device inputs raise it.
    from mxalloy.runtime import ComponentSpec, detect_device_profile

    with pytest.raises(ConfigurationError):
        detect_device_profile(memory_budget_gb=-1.0)
    with pytest.raises(ConfigurationError):
        ComponentSpec(name="no-data").memory_gb("int4")
