"""Verify First-Block Caching on Z-Image (the model where it's near-lossless).

FBC is *excluded from FLUX* (there it visibly shifts the image) and used only on models whose
output it doesn't affect. This measures the Z-Image speedup AND the actual pixel-diff vs the
no-cache baseline (the original script only eyeballed it).

Run with:
    PYTHONPATH=. .venv/bin/python experiments/verify_caching.py
"""

from __future__ import annotations

import time

import numpy as np

from mxdiffusers.zimage.pipeline import MXZimagePipeline

PROMPT = "a beautiful alloy sculpture in a futuristic laboratory, cinematic lighting"
SEED, STEPS, SIZE = 42, 8, 512


def run(pipe: MXZimagePipeline, threshold: float):
    pipe(PROMPT, seed=SEED, num_inference_steps=STEPS, height=SIZE, width=SIZE, cache_threshold=threshold)
    t0 = time.perf_counter()
    res = pipe(PROMPT, seed=SEED, num_inference_steps=STEPS, height=SIZE, width=SIZE, cache_threshold=threshold)
    dt = time.perf_counter() - t0
    tr = pipe._engine.transformer
    return dt, np.array(res.images[0]).astype(int), tr.computed_count, tr.skipped_count


def main() -> None:
    pipe = MXZimagePipeline.from_pretrained(quantize_bits=4)
    bt, bi, bc, bs = run(pipe, 0.0)  # baseline (FBC off; exact caption cache still on)
    ct, ci, cc, cs = run(pipe, 0.25)  # default FBC threshold
    diff = np.abs(bi - ci)
    print(f"baseline     {bt:6.1f}s  computed/skipped {bc}/{bs}")
    print(f"cached_0.25  {ct:6.1f}s  computed/skipped {cc}/{cs}  speedup {bt / ct:.2f}x")
    print(f"pixel diff vs baseline: mean {diff.mean():.2f}  max {int(diff.max())} /255")


if __name__ == "__main__":
    main()
