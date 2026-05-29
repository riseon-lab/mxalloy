"""Alloy exception hierarchy.

All Alloy-specific errors derive from :class:`AlloyError`, so callers can catch the whole
family with ``except AlloyError``. Where a failure is also naturally a builtin error (a bad
value, a bad runtime state), the specific type multiply-inherits from that builtin, so
existing ``except ValueError`` / ``except RuntimeError`` handlers keep working.
"""

from __future__ import annotations


class AlloyError(Exception):
    """Base class for all Alloy errors."""


class ConfigurationError(AlloyError, ValueError):
    """Invalid or conflicting Alloy configuration."""


class QuantizationError(AlloyError, ValueError):
    """A weight could not be quantized or dequantized as requested."""


class IncompatibleLoRAError(AlloyError, ValueError):
    """A LoRA adapter is incompatible with the target model."""


class UnsupportedHardwareError(AlloyError, RuntimeError):
    """The current device or runtime cannot run the requested path."""


class ModelLoadError(AlloyError, RuntimeError):
    """A model or its weights could not be located or loaded."""
