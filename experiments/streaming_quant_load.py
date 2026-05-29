"""Spike: streaming quantized load vs naive load-then-quantize (peak memory).

mflux loads the full bf16 model and only then quantizes, so it peaks at the full model
size (the root cause of klein's ~17.9GB peak). Streaming quantizes each tensor as it
loads and frees the bf16 immediately, peaking near steady-state.

This measures the gap on klein's transformer (the largest single component, ~7.75GB bf16).
Relies on mx.load being lazy: tensors aren't materialized until evaluated.

    .venv/bin/python experiments/streaming_quant_load.py [--bits 4]
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import mlx.core as mx

from mxalloy.runtime.loader import StreamingQuantConfig, load_quantized

GROUP_SIZE = 64


def find_transformer() -> str:
    pattern = str(
        Path.home()
        / ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B"
        / "snapshots/*/transformer/*.safetensors"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("klein transformer not found in HF cache")
    return files[0]


def quantizable(w: mx.array) -> bool:
    return w.ndim == 2 and w.shape[-1] % GROUP_SIZE == 0


def gb(n: int) -> float:
    return round(n / 1e9, 3)


def _nbytes(v: object) -> int:
    if isinstance(v, (list, tuple)):
        return sum(x.nbytes for x in v)
    return v.nbytes  # type: ignore[attr-defined]


def run_naive(path: str, bits: int) -> dict:
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    weights = mx.load(path)
    mx.eval(list(weights.values()))  # materialize the full bf16 model first
    after_load = mx.get_active_memory()
    quant: dict[str, object] = {}
    for k, w in weights.items():
        quant[k] = mx.quantize(w, group_size=GROUP_SIZE, bits=bits) if quantizable(w) else w
    flat: list[mx.array] = []
    for v in quant.values():
        flat.extend(v) if isinstance(v, (list, tuple)) else flat.append(v)  # type: ignore[arg-type]
    mx.eval(flat)
    peak = mx.get_peak_memory()
    size = sum(_nbytes(v) for v in quant.values())
    return {
        "after_load_gb": gb(after_load),
        "peak_gb": gb(peak),
        "quantized_gb": gb(size),
        "secs": round(time.perf_counter() - t0, 2),
    }


def run_streaming(path: str, bits: int) -> dict:
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    quant = load_quantized(path, StreamingQuantConfig(bits=bits, group_size=GROUP_SIZE))
    peak = mx.get_peak_memory()
    n_quantized = sum(1 for v in quant.values() if isinstance(v, (list, tuple)))
    size = sum(_nbytes(v) for v in quant.values())
    return {
        "peak_gb": gb(peak),
        "quantized_gb": gb(size),
        "n_quantized": n_quantized,
        "n_total": len(quant),
        "secs": round(time.perf_counter() - t0, 2),
    }


def find_all_safetensors() -> list[str]:
    base = (
        Path.home()
        / ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B"
        / "snapshots"
    )
    files: list[str] = []
    for component in ("transformer", "text_encoder", "vae"):
        files += sorted(glob.glob(str(base / "*" / component / "*.safetensors")))
    if not files:
        raise SystemExit("klein components not found in HF cache")
    return files


def run_streaming_full(paths: list[str], bits: int) -> dict:
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    cfg = StreamingQuantConfig(bits=bits, group_size=GROUP_SIZE)
    quant: dict[str, object] = {}
    for p in paths:
        for k, v in load_quantized(p, cfg).items():
            quant[f"{p}:{k}"] = v
    peak = mx.get_peak_memory()
    n_quantized = sum(1 for v in quant.values() if isinstance(v, (list, tuple)))
    size = sum(_nbytes(v) for v in quant.values())
    return {
        "peak_gb": gb(peak),
        "quantized_gb": gb(size),
        "n_quantized": n_quantized,
        "n_total": len(quant),
        "secs": round(time.perf_counter() - t0, 2),
    }


# mflux's full-model load-then-quantize peak, measured earlier on this machine.
MFLUX_FULL_PEAK_GB = 17.94


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4, choices=[3, 4, 6, 8])
    ap.add_argument("--full", action="store_true", help="stream all components")
    args = ap.parse_args()

    if args.full:
        paths = find_all_safetensors()
        print(f"full model: {len(paths)} files  bits={args.bits}  gs={GROUP_SIZE}")
        s = run_streaming_full(paths, args.bits)
        print(
            f"streaming full-model peak: {s['peak_gb']} GB "
            f"(quant {s['quantized_gb']} GB, {s['n_quantized']}/{s['n_total']}, {s['secs']}s)"
        )
        print(f"mflux full-model load+quantize peak: {MFLUX_FULL_PEAK_GB} GB")
        if s["peak_gb"]:
            print(f"vs mflux: {round(MFLUX_FULL_PEAK_GB / s['peak_gb'], 2)}x lower peak")
        return

    path = find_transformer()
    print(f"transformer: {Path(path).name}  bits={args.bits}  group_size={GROUP_SIZE}")

    naive = run_naive(path, args.bits)
    streaming = run_streaming(path, args.bits)

    print()
    print(f"{'mode':<12}{'peak GB':>10}{'quant GB':>10}{'secs':>8}")
    print(f"{'naive':<12}{naive['peak_gb']:>10}{naive['quantized_gb']:>10}{naive['secs']:>8}")
    print(
        f"{'streaming':<12}{streaming['peak_gb']:>10}{streaming['quantized_gb']:>10}{streaming['secs']:>8}"
    )
    print()
    print(f"naive after-load (full bf16): {naive['after_load_gb']} GB")
    print(f"streaming quantized {streaming['n_quantized']}/{streaming['n_total']} tensors")
    if streaming["peak_gb"]:
        print(f"peak reduction: {round(naive['peak_gb'] / streaming['peak_gb'], 2)}x")


if __name__ == "__main__":
    main()
