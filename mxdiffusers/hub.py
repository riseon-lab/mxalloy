"""Resolve ``from_pretrained`` model ids to local checkpoint directories.

mxdiffusers is offline-first: it never downloads weights itself. ``resolve_model_dir`` accepts
either a local checkpoint directory or a Hugging Face repo id (resolved against the local HF
cache) and raises :class:`mxalloy.errors.ModelLoadError` with the exact download command when
the checkpoint is absent. mlx-free.
"""

from __future__ import annotations

import glob
from pathlib import Path

from mxalloy.errors import ModelLoadError


def resolve_model_dir(model_id: str | None, *, default_repo: str) -> str:
    """Resolve ``model_id`` (local dir, HF repo id, or None for ``default_repo``) to a dir.

    A Hugging Face repo id is looked up in the local HF hub cache
    (``~/.cache/huggingface/hub``); the newest snapshot wins. Raises ``ModelLoadError`` when
    nothing resolves, with the ``huggingface-cli download`` command to fix it.
    """
    if model_id is not None and Path(model_id).is_dir():
        return model_id
    repo = model_id if model_id is not None else default_repo
    cache_name = "models--" + repo.replace("/", "--")
    pattern = str(Path.home() / ".cache/huggingface/hub" / cache_name / "snapshots/*")
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise ModelLoadError(
            f"{repo} not found locally (not a directory, and not in the Hugging Face cache). "
            f"Download it first: huggingface-cli download {repo}"
        )
    return dirs[-1]
