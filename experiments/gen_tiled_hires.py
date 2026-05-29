"""Generate at HD / 2048**2 with the default (tiled) engine and report peak memory.

Confirms tiled VAE decode lets the resident 18 GB engine render resolutions that full
decode could not (full-decode peak was predicted >24 GB at HD, >44 GB at 2048**2).
Resident engine; peak is reset per resolution. Saves each image for visual seam inspection.

    PYTHONPATH=. .venv/bin/python experiments/gen_tiled_hires.py
"""

from __future__ import annotations

import time

import mlx.core as mx

from mxalloy.models.flux2.engine import Flux2KleinEngine

PROMPT = "a brushed alloy sculpture under studio light"
SEED = 42
RESOLUTIONS = [(1920, 1080), (1080, 1920), (2048, 2048)]  # (width, height)


def gb(n: int) -> float:
    return round(n / 1e9, 2)


def main() -> None:
    engine = Flux2KleinEngine(quantize_bits=4)  # default vae_tile_latent=128
    print(f"load peak {gb(mx.get_peak_memory())} GB  tile_latent={engine.vae_tile_latent}", flush=True)
    for w, h in RESOLUTIONS:
        mx.clear_cache()
        mx.reset_peak_memory()
        t0 = time.time()
        img = engine.generate(PROMPT, seed=SEED, steps=4, height=h, width=w)
        dt = time.time() - t0
        peak = gb(mx.get_peak_memory())
        path = f"experiments/hires_{w}x{h}.png"
        img.save(path)
        print(f"{w}x{h}  4 steps: {dt:.1f}s  peak {peak} GB  -> {path} {img.size}", flush=True)


if __name__ == "__main__":
    main()
