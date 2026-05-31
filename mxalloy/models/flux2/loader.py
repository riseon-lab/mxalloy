"""FLUX.2-klein checkpoint location (model-specific).

The streaming quantized load itself is model-agnostic and lives in the core
(``mxalloy.load_quantized`` / ``mxalloy.component_files``). Only the klein-specific cache
path lives here. INTERNAL.
"""

from __future__ import annotations

import glob
from pathlib import Path


def find_klein_model_dir() -> str:
    pattern = str(
        Path.home()
        / ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B"
        / "snapshots/*"
    )
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError("FLUX.2-klein-4B not found in the Hugging Face cache")
    return dirs[-1]
