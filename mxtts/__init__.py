"""mxtts - speech pipelines that run on or alongside the mxalloy runtime.

The top-level package stays import-light. Family implementations may pull in MLX, Torch, or
codec libraries only when their pipeline is actually constructed.
"""

from mxtts.pipeline import MXAudioResult, MXTTSPipeline

_LAZY = {
    "MXMisoTTSPipeline": "mxtts.miso.pipeline",
}

__all__ = ["MXAudioResult", "MXTTSPipeline", "MXMisoTTSPipeline"]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
