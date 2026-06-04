"""Split the klein generation peak into phases (encode / denoise / decode).

Tells us which phase drives the gen peak at a given resolution, so we know whether tiled
VAE decode alone can unlock higher resolutions on 18GB or whether the transformer denoise
needs activation management too. Reuses the resident engine.

    PYTHONPATH=. .venv/bin/python experiments/profile_peak.py [--height 1024 --width 1024]
"""

from __future__ import annotations

import argparse

import mlx.core as mx

from mxdiffusers.flux.engine import Flux2KleinEngine
from mxdiffusers.flux.latents import prepare_packed_latents, prepare_text_ids
from mxdiffusers.flux.scheduler import FlowMatchEulerScheduler

PROMPT = "a brushed alloy sculpture under studio light"
SEED = 42


def gb(n: int) -> float:
    return round(n / 1e9, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4)
    args = ap.parse_args()
    h, w, steps = args.height, args.width, args.steps

    mx.reset_peak_memory()
    engine = Flux2KleinEngine(quantize_bits=4)
    load_peak = gb(mx.get_peak_memory())
    print(f"resolution {w}x{h}  steps {steps}")
    print(f"load peak: {load_peak} GB")

    # --- encode ---
    mx.clear_cache()
    mx.reset_peak_memory()
    input_ids, attention_mask = engine.tokenizer.encode(PROMPT)
    prompt_embeds = engine.text_encoder.get_prompt_embeds(input_ids, attention_mask, (9, 18, 27))
    mx.eval(prompt_embeds)
    print(f"encode peak:  {gb(mx.get_peak_memory())} GB")

    # --- denoise ---
    mx.clear_cache()
    mx.reset_peak_memory()
    latents, latent_ids, lh, lw = prepare_packed_latents(seed=SEED, height=h, width=w, batch_size=1)
    text_ids = prepare_text_ids(prompt_embeds)
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
    print(f"denoise peak: {gb(mx.get_peak_memory())} GB")

    # --- decode ---
    mx.clear_cache()
    mx.reset_peak_memory()
    packed = latents.reshape(1, lh, lw, latents.shape[-1]).transpose(0, 3, 1, 2)
    decoded = engine.vae.decode_packed_latents(packed)
    mx.eval(decoded)
    print(f"decode peak:  {gb(mx.get_peak_memory())} GB   (output {decoded.shape})")


if __name__ == "__main__":
    main()
