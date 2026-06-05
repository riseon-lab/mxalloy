from __future__ import annotations

from typing import Any

import pytest


def require_mlx_core() -> Any:
    mx = pytest.importorskip("mlx.core")
    try:
        mx.eval(mx.array([0]))
    except RuntimeError as exc:
        if "No Metal device available" in str(exc):
            pytest.skip(str(exc), allow_module_level=True)
        raise
    return mx


def require_mlx_nn() -> tuple[Any, Any, Any]:
    mx = require_mlx_core()
    try:
        from mlx import nn
        from mlx.utils import tree_flatten
    except RuntimeError as exc:
        if "No Metal device available" in str(exc):
            pytest.skip(str(exc), allow_module_level=True)
        raise
    return mx, nn, tree_flatten
