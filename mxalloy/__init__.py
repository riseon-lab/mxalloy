"""Alloy public package surface."""

from typing import TYPE_CHECKING

from mxalloy.config import AlloyConfig, QuantizationConfig, RuntimeConfig
from mxalloy.errors import (
    AlloyError,
    ConfigurationError,
    IncompatibleLoRAError,
    ModelLoadError,
    QuantizationError,
    UnsupportedHardwareError,
)

# The core memory primitive. Exposed lazily so `import mxalloy` stays mlx-free; the loader
# (and its mlx dependency) is imported only when first accessed.
_LAZY = {"load_quantized", "QuantConfig", "component_files"}

if TYPE_CHECKING:
    from mxalloy.loader import QuantConfig, component_files, load_quantized

__all__ = [
    "AlloyConfig",
    "AlloyError",
    "ConfigurationError",
    "IncompatibleLoRAError",
    "ModelLoadError",
    "QuantConfig",
    "QuantizationConfig",
    "QuantizationError",
    "RuntimeConfig",
    "UnsupportedHardwareError",
    "component_files",
    "load_quantized",
]


def __getattr__(name: str):  # PEP 562: lazy, mlx-only-on-use
    if name in _LAZY:
        from mxalloy import loader

        return getattr(loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
