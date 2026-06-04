"""mxdiffusers — a diffusers-style diffusion framework for Apple Silicon, running on mxalloy.

Model-family pipelines (e.g. ``MXFluxPipeline``) build on the shared ``MXPipeline`` base and
delegate device detection, memory planning, precision selection, and quantized loading to the
mxalloy runtime. mxdiffusers depends on mxalloy; mxalloy never depends on mxdiffusers.

See ``PROVENANCE.md`` for the lineage of individual model implementations.
"""

from mxdiffusers.pipeline import MXPipeline, MXResult

# Family pipelines pull in mlx (via their model graph), so expose them lazily — importing
# ``mxdiffusers`` (and the MXPipeline base) stays light.
_LAZY = {"MXFluxPipeline": "mxdiffusers.flux.pipeline"}

__all__ = ["MXPipeline", "MXResult", "MXFluxPipeline"]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
