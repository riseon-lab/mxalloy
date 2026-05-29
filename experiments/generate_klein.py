"""Generate a klein image end-to-end with mxalloy + measure peak memory.

The payoff: mxalloy generating FLUX.2-klein natively (no mflux), resident, at low peak.
Compare the output to the mflux image at the same seed/prompt (experiments/klein_4bit_spike.png).

    PYTHONPATH=. .venv/bin/python experiments/generate_klein.py
"""

import time

import mlx.core as mx

from mxalloy.models.flux2.engine import Flux2KleinEngine

PROMPT = "a brushed alloy sculpture under studio light"
SEED = 42


def gb(n: int) -> float:
    return round(n / 1e9, 2)


def main() -> None:
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    engine = Flux2KleinEngine(quantize_bits=4)
    print(
        f"loaded in {time.perf_counter() - t0:.1f}s  peak-after-load={gb(mx.get_peak_memory())} GB"
    )

    t1 = time.perf_counter()
    image = engine.generate(PROMPT, seed=SEED, steps=4)
    image.save("experiments/klein_mxalloy.png")
    print(f"generated in {time.perf_counter() - t1:.1f}s  peak={gb(mx.get_peak_memory())} GB")
    print("saved experiments/klein_mxalloy.png")


if __name__ == "__main__":
    main()
