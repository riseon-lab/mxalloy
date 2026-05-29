"""Benchmark harness for Alloy FLUX inference.

For now this captures the run environment (device, MLX version, Python version) and a
small synthetic timing, so the harness is proven before real FLUX numbers exist.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from mxalloy.runtime import detect_device


def _mlx_version() -> str | None:
    try:
        return version("mlx")
    except PackageNotFoundError:
        return None


def _smoke_timing() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((1024, 1024)).astype(np.float32)
    b = rng.standard_normal((1024, 1024)).astype(np.float32)
    iterations = 5
    checksum = 0.0
    started = time.perf_counter()
    for _ in range(iterations):
        checksum += float((a @ b).sum())
    elapsed = time.perf_counter() - started
    return {
        "op": "matmul_1024x1024",
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "checksum": checksum,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Alloy FLUX generation.")
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark_flux.json"))
    parser.add_argument("--variant", default="fp16-baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = detect_device()
    result: dict[str, Any] = {
        "variant": args.variant,
        "device": {
            "machine": device.machine,
            "processor": device.processor,
            "is_apple_silicon": device.is_apple_silicon,
        },
        "python_version": platform.python_version(),
        "mlx_version": _mlx_version(),
    }
    if args.variant == "smoke":
        result["timing"] = _smoke_timing()
    else:
        result["status"] = "not_implemented"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
