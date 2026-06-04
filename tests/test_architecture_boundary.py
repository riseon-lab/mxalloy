"""Architectural invariant: the mxalloy runtime never depends on the model layer.

mxalloy is reusable Apple-Silicon infrastructure (streaming loader, device/runtime, attention
primitives, quantization). Diffusion model implementations live in ``mxdiffusers``, which
imports mxalloy — never the reverse. Keeping ``mxalloy ↛ mxdiffusers`` one-directional is what
lets mxalloy ship as a clean, model-free (and mflux-free) package. This test is the
enforcement: it parses every mxalloy module and fails on any import of ``mxdiffusers``.
"""

from __future__ import annotations

import ast
from pathlib import Path

_MXALLOY = Path(__file__).resolve().parents[1] / "mxalloy"


def _py_files() -> list[Path]:
    return sorted(_MXALLOY.rglob("*.py"))


def _imports_prefix(path: Path, prefix: str) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == prefix or a.name.startswith(prefix + ".") for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == prefix or module.startswith(prefix + "."):
                return True
    return False


def test_mxalloy_has_files() -> None:
    # Guard against the glob silently matching nothing (which would make the test vacuous).
    assert _py_files(), "no mxalloy files discovered"


def test_mxalloy_never_imports_mxdiffusers() -> None:
    offenders = sorted(
        str(p.relative_to(_MXALLOY)) for p in _py_files() if _imports_prefix(p, "mxdiffusers")
    )
    assert not offenders, f"mxalloy (runtime) must not import mxdiffusers (models): {offenders}"
