from __future__ import annotations

from pathlib import Path

import pytest

from mxalloy.errors import ModelLoadError
from mxdiffusers.hub import resolve_model_dir


def _fake_cache(home: Path, repo: str) -> Path:
    snap = (
        home
        / ".cache/huggingface/hub"
        / ("models--" + repo.replace("/", "--"))
        / "snapshots"
        / "abc123"
    )
    snap.mkdir(parents=True)
    return snap


def test_local_directory_wins(tmp_path) -> None:
    model_dir = tmp_path / "checkpoint"
    model_dir.mkdir()
    assert resolve_model_dir(str(model_dir), default_repo="org/model") == str(model_dir)


def test_hf_repo_id_resolves_from_cache(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    snap = _fake_cache(tmp_path, "org/model")
    assert resolve_model_dir("org/model", default_repo="other/default") == str(snap)


def test_none_resolves_default_repo(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    snap = _fake_cache(tmp_path, "org/default")
    assert resolve_model_dir(None, default_repo="org/default") == str(snap)


def test_newest_snapshot_wins(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    snap = _fake_cache(tmp_path, "org/model")
    newer = snap.parent / "def456"
    newer.mkdir()
    assert resolve_model_dir("org/model", default_repo="org/model") == str(newer)


def test_missing_model_raises_model_load_error_with_hint(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    with pytest.raises(ModelLoadError, match="huggingface-cli download org/absent"):
        resolve_model_dir("org/absent", default_repo="org/absent")
