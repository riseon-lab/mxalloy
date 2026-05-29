"""Sweep mxalloy klein runtime + peak memory across resolutions and step counts.

Loads the engine once (resident) and reuses it for every config — the warm path. For each
(resolution, steps) it records wall-clock and peak memory. The engine tiles VAE decode by
default (``--vae-tile`` latent px; ``0`` disables), which caps the decode peak so it
plateaus rather than scaling with pixels — the predictor below accounts for that. Configs
whose *predicted* peak exceeds ``--budget-gb`` are skipped, so we don't thrash swap on a
constrained machine; raise ``--budget-gb`` on a bigger machine to fill them in. Results
stream to JSON so a crash or OOM-kill still preserves everything completed so far.

    PYTHONPATH=. .venv/bin/python benchmarks/benchmark_klein.py --budget-gb 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from mxalloy.models.flux2.engine import Flux2KleinEngine

RESOLUTIONS = [  # (width, height)
    (512, 512),
    (896, 512),  # 16:9 landscape (small)
    (512, 896),  # 9:16 portrait (small)
    (1024, 1024),
    (1344, 768),  # 16:9 landscape (~1MP)
    (768, 1344),  # 9:16 portrait (~1MP)
    (1080, 1920),  # HD portrait
    (1920, 1080),  # HD landscape
    (2048, 2048),
]
STEPS = [4, 20, 50]
PROMPT = "a brushed alloy sculpture under studio light"
SEED = 42


def gb(n: int) -> float:
    return round(n / 1e9, 2)


def _recommended_gb() -> float:
    try:
        return mx.device_info().get("max_recommended_working_set_size", 0) / 1e9
    except Exception:  # noqa: BLE001
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantize", type=int, default=4)
    ap.add_argument("--budget-gb", type=float, default=16.0)
    ap.add_argument(
        "--vae-tile", type=int, default=128, help="VAE decode tile (latent px); 0 disables tiling"
    )
    ap.add_argument("--output", default="experiments/benchmark_klein.json")
    ap.add_argument(
        "--resolutions", default=None, help="comma-separated WxH (default: built-in set)"
    )
    ap.add_argument("--steps", default=None, help="comma-separated step counts (default: 4,20,50)")
    args = ap.parse_args()

    resolutions = (
        [tuple(int(v) for v in r.lower().split("x")) for r in args.resolutions.split(",")]
        if args.resolutions
        else RESOLUTIONS
    )
    steps_list = [int(s) for s in args.steps.split(",")] if args.steps else STEPS

    print(
        f"device recommended working set: {_recommended_gb():.1f} GB  budget: {args.budget_gb} GB"
        f"  vae_tile_latent: {args.vae_tile or None}"
    )

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    engine = Flux2KleinEngine(quantize_bits=args.quantize, vae_tile_latent=args.vae_tile or None)
    load_time = round(time.perf_counter() - t0, 1)
    load_peak = gb(mx.get_peak_memory())
    print(f"loaded in {load_time}s  load peak {load_peak} GB")

    results: dict = {
        "quantize_bits": args.quantize,
        "vae_tile_latent": args.vae_tile or None,
        "budget_gb": args.budget_gb,
        "load_time_s": load_time,
        "load_peak_gb": load_peak,
        "runs": [],
    }
    observed: list[tuple[int, float]] = []
    # Tiled decode caps activations at one (tile*8)px tile, so peak grows with pixels only
    # up to that tile area, then plateaus. Clamp the pixel term to the tile area so the
    # predictor stops extrapolating linearly above it (else it wrongly skips large configs).
    tile_px = (args.vae_tile * 8) ** 2 if args.vae_tile else None

    def predict(pixels: int) -> float:
        if not observed:
            return 0.0
        eff = (lambda p: min(p, tile_px)) if tile_px else (lambda p: p)
        px_max, peak_max = max(observed, key=lambda o: o[0])
        denom = eff(px_max)
        slope = (peak_max - load_peak) / denom if denom else 0.0
        return load_peak + slope * eff(pixels)

    configs = sorted(
        [(w, h, s) for (w, h) in resolutions for s in steps_list],
        key=lambda c: (c[0] * c[1], c[2]),
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for w, h, steps in configs:
        pixels = w * h
        predicted = predict(pixels)
        if predicted > args.budget_gb:
            row = {
                "width": w,
                "height": h,
                "steps": steps,
                "status": "skipped",
                "predicted_peak_gb": round(predicted, 1),
            }
            print(
                f"{w}x{h} {steps:>2} steps: SKIP (predicted ~{predicted:.1f} GB > {args.budget_gb})"
            )
        else:
            mx.clear_cache()
            mx.reset_peak_memory()
            try:
                t1 = time.perf_counter()
                engine.generate(PROMPT, seed=SEED, steps=steps, height=h, width=w)
                dt = round(time.perf_counter() - t1, 1)
                peak = gb(mx.get_peak_memory())
                observed.append((pixels, peak))
                row = {
                    "width": w,
                    "height": h,
                    "steps": steps,
                    "time_s": dt,
                    "peak_gb": peak,
                    "status": "ok",
                }
                print(f"{w}x{h} {steps:>2} steps: {dt}s  peak {peak} GB")
            except Exception as exc:  # noqa: BLE001
                row = {
                    "width": w,
                    "height": h,
                    "steps": steps,
                    "status": "error",
                    "error": str(exc)[:160],
                }
                print(f"{w}x{h} {steps:>2} steps: ERROR {str(exc)[:80]}")
        results["runs"].append(row)
        out_path.write_text(json.dumps(results, indent=2))

    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
