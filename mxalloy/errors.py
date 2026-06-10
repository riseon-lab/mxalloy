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


class ModelLoadError(AlloyError, RuntimeError):
    """A model or its weights could not be located or loaded."""
