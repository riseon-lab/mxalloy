"""FLUX.2-klein checkpoint location (model-specific).

The streaming quantized load itself is model-agnostic and lives in the core
(``mxalloy.load_quantized`` / ``mxalloy.component_files``). Only the klein-specific cache
path lives here. INTERNAL.
"""

from __future__ import annotations

from mxdiffusers.hub import resolve_model_dir

_KLEIN_REPO = "black-forest-labs/FLUX.2-klein-4B"


def find_klein_model_dir(model_id: str | None = None) -> str:
    """Resolve ``model_id`` (local dir, HF repo id, or None) to a klein checkpoint dir."""
    return resolve_model_dir(model_id, default_repo=_KLEIN_REPO)
