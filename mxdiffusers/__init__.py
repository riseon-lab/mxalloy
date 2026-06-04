"""mxdiffusers — a diffusers-style diffusion framework for Apple Silicon, running on mxalloy.

Model-family pipelines (e.g. ``MXFluxPipeline``) build on a shared ``MXPipeline`` base and
delegate device detection, memory planning, precision selection, and quantized loading to the
mxalloy runtime. mxdiffusers depends on mxalloy; mxalloy never depends on mxdiffusers.

See ``PROVENANCE.md`` for the lineage of individual model implementations.
"""
