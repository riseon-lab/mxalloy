from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def _run(code: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(_REPO)}
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)


def test_import_mxalloy_is_mlx_free() -> None:
    # `import mxalloy` must not pull in mlx: the core stays import-light, and mlx loads only
    # when a runtime primitive (e.g. load_quantized) is first accessed. Checked in a fresh
    # subprocess so it is independent of whatever other tests have already imported.
    result = _run(
        "import mxalloy, sys; "
        "leaked = sorted(m for m in sys.modules if m == 'mlx' or m.startswith('mlx.')); "
        "assert not leaked, leaked"
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_import_mxtts_is_heavy_dependency_free() -> None:
    result = _run(
        "import mxtts, sys; "
        "leaked = sorted("
        "m for m in sys.modules "
        "if m in {'mlx', 'torch', 'torchaudio'} "
        "or m.startswith(('mlx.', 'torch.', 'torchaudio.'))"
        "); "
        "assert not leaked, leaked"
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_import_miso_pipeline_is_heavy_dependency_free() -> None:
    result = _run(
        "from mxtts import MXMisoTTSPipeline; import sys; "
        "leaked = sorted("
        "m for m in sys.modules "
        "if m in {'mlx', 'torch', 'torchaudio'} "
        "or m.startswith(('mlx.', 'torch.', 'torchaudio.'))"
        "); "
        "assert MXMisoTTSPipeline.family == 'miso'; "
        "assert not leaked, leaked"
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_public_symbols_are_exported() -> None:
    import mxalloy

    # Always importable with no mlx dependency.
    for name in (
        "AlloyConfig",
        "QuantizationConfig",
        "RuntimeConfig",
        "AlloyError",
        "ConfigurationError",
    ):
        assert hasattr(mxalloy, name), name
    # mlx-backed core loader: exposed lazily (resolving these imports mlx), so assert via
    # __all__ rather than forcing the import here.
    for name in ("QuantConfig", "load_quantized", "component_files"):
        assert name in mxalloy.__all__, name


def test_detect_device_returns_structured_result() -> None:
    from mxalloy.runtime import DeviceProfile, WorkloadSpec, detect_device, plan_execution

    device = detect_device()
    assert isinstance(device.machine, str)
    assert isinstance(device.is_apple_silicon, bool)
    assert DeviceProfile
    assert WorkloadSpec
    assert plan_execution
