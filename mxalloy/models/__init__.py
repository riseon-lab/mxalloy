"""Model adapters.

Models consume the reusable core (``mxalloy.load_quantized``, ``mxalloy.runtime``); the core
never imports from here — see ``tests/test_architecture_boundary.py``, which enforces that
boundary. Concrete adapters live in subpackages, e.g. ``mxalloy.models.flux2``.
"""
