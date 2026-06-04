"""Verify tiled VAE decode: quality (vs full decode) and peak memory.

Decodes one real denoised latent two ways -- full and tiled -- and reports the pixel
difference (incl. a seam probe) plus the decode peak for each. Confirms tiling reduces the
decode peak without visible seams, and that a latent fitting one tile is bit-exact.

    PYTHONPATH=. .venv/bin/python experiments/verify_tiled_vae.py \
        [--height 1024 --width 1024 --tile 64]
"""

from __future__ import annotations

import argparse

import mlx.core as mx
import numpy as np
from PIL import Image

from mxdiffusers.flux.engine import Flux2KleinEngine
from mxdiffusers.flux.latents import prepare_packed_latents, prepare_text_ids
from mxdiffusers.flux.scheduler import FlowMatchEulerScheduler

PROMPT = "a brushed alloy sculpture under studio light"
SEED = 42


def gb(n: int) -> float:
    return round(n / 1e9, 2)


def to_uint8(decoded: mx.array) -> np.ndarray:
    x = mx.clip(decoded / 2 + 0.5, 0, 1)
    x = mx.transpose(x, (0, 2, 3, 1)).astype(mx.float32)
    return (np.array(x) * 255).round().astype(np.uint8)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--tile", type=int, default=64)
    args = ap.parse_args()
    h, w, steps, tile = args.height, args.width, args.steps, args.tile

    engine = Flux2KleinEngine(quantize_bits=4)

    # --- denoise once to get a real latent ---
    input_ids, attention_mask = engine.tokenizer.encode(PROMPT)
    prompt_embeds = engine.text_encoder.get_prompt_embeds(input_ids, attention_mask, (9, 18, 27))
    text_ids = prepare_text_ids(prompt_embeds)
    latents, latent_ids, lh, lw = prepare_packed_latents(seed=SEED, height=h, width=w, batch_size=1)
    scheduler = FlowMatchEulerScheduler(
        num_inference_steps=steps, image_seq_len=(h // 16) * (w // 16)
    )
    for t in range(steps):
        noise = engine.transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            timestep=scheduler.timesteps[t],
            img_ids=latent_ids,
            txt_ids=text_ids,
            guidance=None,
        )
        latents = scheduler.step(noise, t, latents)
        mx.eval(latents)
    packed = latents.reshape(1, lh, lw, latents.shape[-1]).transpose(0, 3, 1, 2)
    mx.eval(packed)

    print(f"resolution {w}x{h}  steps {steps}  tile_latent {tile}")

    # --- full decode ---
    mx.clear_cache()
    mx.reset_peak_memory()
    full = engine.vae.decode_packed_latents(packed, tile_latent=None)
    mx.eval(full)
    full_peak = gb(mx.get_peak_memory())
    full_img = to_uint8(full)
    del full

    # --- tiled decode ---
    mx.clear_cache()
    mx.reset_peak_memory()
    tiled = engine.vae.decode_packed_latents(packed, tile_latent=tile)
    mx.eval(tiled)
    tiled_peak = gb(mx.get_peak_memory())
    tiled_img = to_uint8(tiled)
    del tiled

    print(f"full  decode peak: {full_peak} GB")
    print(f"tiled decode peak: {tiled_peak} GB")

    d = np.abs(full_img.astype(np.int16) - tiled_img.astype(np.int16))
    print(f"pixel diff: max {int(d.max())}  mean {d.mean():.4f}  >2: {(d > 2).mean() * 100:.3f}%")

    # seam probe: a tile boundary would show as a bright row/col line in the diff.
    row = d.mean(axis=(1, 2))
    col = d.mean(axis=(0, 2))
    print(
        f"seam probe: worst row {row.max():.3f} @ y={int(row.argmax())}  "
        f"worst col {col.max():.3f} @ x={int(col.argmax())}"
    )

    Image.fromarray(full_img).save(f"experiments/tiled_full_{w}x{h}.png")
    Image.fromarray(tiled_img).save(f"experiments/tiled_tile{tile}_{w}x{h}.png")


if __name__ == "__main__":
    main()
