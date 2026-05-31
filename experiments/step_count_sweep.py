"""klein step-count vs time/quality: is 4 steps necessary, or can we go fewer (faster)?

Generates the same prompt/seed at increasing step counts, saves each image (eyeball quality),
times each, and reports how far each is from the highest-step "converged" reference -- once
consecutive step counts stop changing the image, more steps are wasted wall-clock.

    PYTHONPATH=. .venv/bin/python experiments/step_count_sweep.py [--height 1024 --width 1024]
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np
from PIL import Image

from mxalloy.models.flux2.engine import Flux2KleinEngine

PROMPT = "a brushed alloy sculpture under studio light"
SEED = 42
STEPS = [1, 2, 3, 4, 6, 8]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    args = ap.parse_args()
    h, w = args.height, args.width

    engine = Flux2KleinEngine(quantize_bits=4)
    imgs: dict[int, np.ndarray] = {}
    print(f"{w}x{h}  prompt={PROMPT!r}  seed={SEED}", flush=True)
    for steps in STEPS:
        mx.clear_cache()
        t = time.time()
        img = engine.generate(PROMPT, seed=SEED, steps=steps, height=h, width=w)
        dt = time.time() - t
        path = f"experiments/steps_{steps}_{w}x{h}.png"
        img.save(path)
        imgs[steps] = np.asarray(img).astype(np.int16)
        print(f"{steps:>2} steps: {dt:5.1f}s  ({dt/steps:4.1f}s/step)  -> {path}", flush=True)

    ref = imgs[STEPS[-1]]  # highest step count = "converged" reference
    print(f"\nconvergence (mean abs px diff vs {STEPS[-1]}-step):", flush=True)
    for steps in STEPS[:-1]:
        d = np.abs(imgs[steps] - ref)
        print(f"  {steps:>2} steps: mean {d.mean():5.2f}/255  max {int(d.max()):3d}", flush=True)


if __name__ == "__main__":
    main()
