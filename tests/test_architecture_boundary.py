"""Architectural invariant: the reusable core never imports model adapters.

mxalloy is positioned as model-agnostic infrastructure. ``mxalloy/models`` consumes the core
(``load_quantized``, ``runtime``, ``attention``); the core must never import back into
``mxalloy.models``. Keeping this one-directional makes the public API boundary real and lets
the core ship without any particular model. This test is the enforcement (a lightweight
import-linter): it parses each core module and fails on any ``mxalloy.models`` import.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1] / "mxalloy"

# The reusable core: top-level modules + model-agnostic subpackages. Everything here must be
# usable with no model adapter present.
_CORE = (
    "__init__.py",
    "loader.py",
    "config.py",
    "errors.py",
    "runtime",
    "attention",
    "kernels",
    "utils",
)


def _core_files() -> list[Path]:
    files: list[Path] = []
    for entry in _CORE:
        path = _PKG / entry
        if path.is_dir():
            files.extend(path.rglob("*.py"))
        elif path.exists():
            files.append(path)
    return files


def _imports_models(path: Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "mxalloy.models" or alias.name.startswith("mxalloy.models."):
                    return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "mxalloy.models" or module.startswith("mxalloy.models."):
                return True
    return False


def test_core_files_exist() -> None:
    # Guard against the glob matching nothing (which would make the boundary test vacuous).
    assert _core_files(), "no core files discovered"


def test_core_never_imports_models() -> None:
    offenders = sorted(
        str(p.relative_to(_PKG)) for p in _core_files() if _imports_models(p)
    )
    assert not offenders, f"core modules must not import mxalloy.models: {offenders}"
