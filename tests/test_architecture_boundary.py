"""Architectural invariant: the mxalloy runtime never depends on the model layer.

mxalloy is reusable Apple-Silicon infrastructure (streaming loader, device/runtime, attention
primitives, quantization). Model implementations live in sibling packages such as
``mxdiffusers`` and ``mxtts``, which import mxalloy - never the reverse. Keeping mxalloy
model-free is what lets it ship as clean reusable infrastructure. This test is the enforcement:
it parses every mxalloy module and fails on any import of a model package.
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


def test_mxalloy_never_imports_model_packages() -> None:
    prefixes = ("mxdiffusers", "mxtts")
    offenders = sorted(
        f"{p.relative_to(_MXALLOY)} imports {prefix}"
        for p in _py_files()
        for prefix in prefixes
        if _imports_prefix(p, prefix)
    )
    assert not offenders, f"mxalloy (runtime) must not import model packages: {offenders}"
