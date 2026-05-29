"""Profile where per-step denoise time goes: attention's share, and mx.compile upside.

The proposed next perf milestone targets per-step transformer time via "attention tiling".
Before committing, measure attention's actual share: MLX's fast SDPA is already a fused
flash-attention kernel, so the real lever may be the quantized GEMMs / op fusion, not
hand-rolled tiling. Prints baseline per-step, the pure-SDPA floor (its % of a step), and
the speedup from wrapping the forward in mx.compile.

    PYTHONPATH=. .venv/bin/python experiments/profile_step.py [--height 1024 --width 1024]
"""

from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx
from mlx.core.fast import scaled_dot_product_attention

from mxalloy.models.flux2.engine import Flux2KleinEngine
from mxalloy.models.flux2.latents import prepare_packed_latents, prepare_text_ids
from mxalloy.models.flux2.scheduler import FlowMatchEulerScheduler

PROMPT = "a brushed alloy sculpture under studio light"
SEED = 42


def bench(thunk, iters: int = 3, warmup: int = 1) -> float:
    for _ in range(warmup):
        mx.eval(thunk())
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        mx.eval(thunk())
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    args = ap.parse_args()
    h, w = args.height, args.width

    engine = Flux2KleinEngine(quantize_bits=4)
    tf = engine.transformer

    input_ids, attn_mask = engine.tokenizer.encode(PROMPT)
    prompt_embeds = engine.text_encoder.get_prompt_embeds(input_ids, attn_mask, (9, 18, 27))
    text_ids = prepare_text_ids(prompt_embeds)
    latents, latent_ids, lh, lw = prepare_packed_latents(seed=SEED, height=h, width=w, batch_size=1)
    sched = FlowMatchEulerScheduler(num_inference_steps=4, image_seq_len=(h // 16) * (w // 16))
    t0 = sched.timesteps[0]

    img_tokens = (h // 16) * (w // 16)
    txt_tokens = prompt_embeds.shape[1]
    seq = img_tokens + txt_tokens
    heads, hd = 24, 128
    n_blocks = len(tf.transformer_blocks) + len(tf.single_transformer_blocks)
    print(f"{w}x{h}: seq {seq} ({img_tokens} img + {txt_tokens} txt), {heads}x{hd}, {n_blocks} blocks")

    def fwd(hs, ehs, t):
        return tf(
            hidden_states=hs,
            encoder_hidden_states=ehs,
            timestep=t,
            img_ids=latent_ids,
            txt_ids=text_ids,
            guidance=None,
        )

    step = bench(lambda: fwd(latents, prompt_embeds, t0))
    print(f"baseline per-step:        {step * 1000:8.1f} ms")

    # Pure-SDPA floor: one attention per block at the real shape, no proj/RoPE.
    q = mx.random.normal((1, heads, seq, hd)).astype(mx.bfloat16)
    k = mx.random.normal((1, heads, seq, hd)).astype(mx.bfloat16)
    v = mx.random.normal((1, heads, seq, hd)).astype(mx.bfloat16)
    scale = 1.0 / (hd**0.5)

    def attn_all():
        out = q
        for _ in range(n_blocks):
            out = scaled_dot_product_attention(q, k, v, scale=scale)
        return out

    attn = bench(attn_all, iters=5, warmup=2)
    print(f"pure SDPA x{n_blocks}:           {attn * 1000:8.1f} ms  ({attn / step * 100:4.1f}% of step)")

    try:
        cfwd = mx.compile(fwd)
        cstep = bench(lambda: cfwd(latents, prompt_embeds, t0))
        print(
            f"mx.compile per-step:      {cstep * 1000:8.1f} ms  "
            f"({(1 - cstep / step) * 100:+.0f}% vs baseline)"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"mx.compile failed: {str(exc)[:140]}")


if __name__ == "__main__":
    main()
