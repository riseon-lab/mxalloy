"""mxalloy vs mflux: warm end-to-end klein speed + peak, identical settings.

Same model (FLUX.2-klein-4B), same machine, same config (4-bit, 1024^2, 4 steps, seed 42,
guidance 1.0 -> no CFG, one forward/step for both). Reports warm total generate time + peak
memory. Run one engine per process (--engine) so peak memory is clean and neither warms the
other. mflux is the dev-time reference oracle (never a runtime dep); this answers the question
"is our from-scratch port as fast as the reference?".

Confounder noted: mflux mx.compile()s its transformer forward; mxalloy runs it eagerly. The
as-is numbers are the honest product comparison.

    PYTHONPATH=. .venv/bin/python experiments/bench_vs_mflux.py --engine mflux
    PYTHONPATH=. .venv/bin/python experiments/bench_vs_mflux.py --engine mxalloy
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

PROMPT = "a brushed alloy sculpture under studio light, high detail"
SEED, STEPS = 42, 4
WARMUP, TIMED = 1, 2


def build_mxalloy(size: int):
    from mxalloy.models.flux2.engine import Flux2KleinEngine

    eng = Flux2KleinEngine(quantize_bits=4)
    return lambda: eng.generate(PROMPT, seed=SEED, steps=STEPS, height=size, width=size)


def build_mflux(size: int):
    from mflux.models.common.config import ModelConfig
    from mflux.models.flux2.variants import Flux2Klein

    model = Flux2Klein(quantize=4, model_config=ModelConfig.flux2_klein_4b())
    return lambda: model.generate_image(
        seed=SEED, prompt=PROMPT, num_inference_steps=STEPS, height=size, width=size, guidance=1.0
    )


def gb(n: int) -> float:
    return round(n / 1e9, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["mxalloy", "mflux"], required=True)
    ap.add_argument("--size", type=int, default=1024)
    args = ap.parse_args()

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    gen = build_mxalloy(args.size) if args.engine == "mxalloy" else build_mflux(args.size)
    load_s = time.perf_counter() - t0
    load_peak = gb(mx.get_peak_memory())

    for _ in range(WARMUP):
        gen()  # both return a realized image (forces full eval), warms kernels / mx.compile

    mx.reset_peak_memory()
    mx.clear_cache()
    times = []
    for _ in range(TIMED):
        t = time.perf_counter()
        gen()
        times.append(time.perf_counter() - t)
    gen_peak = gb(mx.get_peak_memory())

    best = min(times)
    print(
        f"\nRESULT [{args.engine} {args.size}^2] load {load_s:.1f}s (peak {load_peak} GB) | "
        f"warm gen min {best:.2f}s mean {sum(times) / len(times):.2f}s "
        f"(~{best / STEPS:.2f}s/step over {STEPS}) | gen peak {gen_peak} GB"
    )


if __name__ == "__main__":
    main()
