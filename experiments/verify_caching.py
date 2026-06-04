"""Benchmark and verify Redundant Projector Caching and First Block Caching on MXAlloy.

Measures load time, warm generation time, speedup factor, step computation stats, 
and saves the output images for visual comparison.

Run with:
    PYTHONPATH=. .venv/bin/python experiments/verify_caching.py
"""

from __future__ import annotations

import os
import time
import mlx.core as mx

from mxdiffusers.flux.pipeline import MXFluxPipeline

PROMPT = "a beautiful alloy sculpture in a futuristic laboratory, cinematic lighting"
SEED = 42
STEPS = 8  # Use 8 steps to give caching more opportunities to trigger
SIZE = 1024


def gb(n: int) -> float:
    return round(n / 1e9, 2)


def run_pipeline(pipe: MXFluxPipeline, cache_threshold: float, name: str) -> tuple[float, float, int, int]:
    print(f"\n--- Running: {name} (cache_threshold={cache_threshold}) ---")
    
    # Warmup
    print("Warming up...")
    pipe(PROMPT, seed=SEED, num_inference_steps=STEPS, height=SIZE, width=SIZE, cache_threshold=cache_threshold)
    
    # Timed runs
    mx.reset_peak_memory()
    mx.clear_cache()
    
    times = []
    iters = 2
    for i in range(iters):
        print(f"Iteration {i + 1}/{iters}...")
        t0 = time.perf_counter()
        res = pipe(PROMPT, seed=SEED, num_inference_steps=STEPS, height=SIZE, width=SIZE, cache_threshold=cache_threshold)
        times.append(time.perf_counter() - t0)
        
    avg_time = sum(times) / len(times)
    peak_mem = gb(mx.get_peak_memory())
    
    # Get stats from transformer
    transformer = pipe._engine.transformer
    computed = getattr(transformer, "computed_count", 0)
    skipped = getattr(transformer, "skipped_count", 0)
    
    # Save image
    os.makedirs("outputs", exist_ok=True)
    out_path = f"outputs/{name.lower().replace(' ', '_')}.png"
    res.images[0].save(out_path)
    print(f"Saved output to {out_path}")
    
    return avg_time, peak_mem, computed, skipped


def main() -> None:
    print(f"Initializing MXFluxPipeline (4-bit, {SIZE}x{SIZE}, {STEPS} steps)...")
    t0 = time.perf_counter()
    pipe = MXFluxPipeline.from_pretrained(quantize_bits=4)
    load_time = time.perf_counter() - t0
    print(f"Loaded in {load_time:.2f}s (peak memory {gb(mx.get_peak_memory())} GB)")

    # 1. Baseline Run (no caching)
    base_time, base_mem, base_comp, base_skip = run_pipeline(pipe, 0.0, "Baseline")
    
    # 2. Caching threshold = 0.15 (conservative/standard)
    cache_time_15, cache_mem_15, comp_15, skip_15 = run_pipeline(pipe, 0.15, "Cached 0.15")
    
    # 3. Caching threshold = 0.25 (more aggressive)
    cache_time_25, cache_mem_25, comp_25, skip_25 = run_pipeline(pipe, 0.25, "Cached 0.25")

    # Print summary table
    print("\n" + "=" * 80)
    print("  MXALLOY CACHING OPTIMIZATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<20} | {'Avg Gen Time':<12} | {'Speedup':<8} | {'Peak Mem':<10} | {'Steps (Comp/Skip)':<20}")
    print("-" * 80)
    print(f"{'Baseline':<20} | {base_time:10.2f}s | {'1.00x':<8} | {base_mem:7.2f} GB | {base_comp}/{base_skip}")
    print(f"{'Cached 0.15':<20} | {cache_time_15:10.2f}s | {base_time/cache_time_15:7.2f}x | {cache_mem_15:7.2f} GB | {comp_15}/{skip_15}")
    print(f"{'Cached 0.25':<20} | {cache_time_25:10.2f}s | {base_time/cache_time_25:7.2f}x | {cache_mem_25:7.2f} GB | {comp_25}/{skip_25}")
    print("=" * 80)
    print("Open the output images in outputs/ to visually verify there is no quality degradation.")


if __name__ == "__main__":
    main()
