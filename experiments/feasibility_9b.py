"""FLUX.2-klein-9B feasibility on 18 GB (milestone D).

Two questions: (1) does the streaming-quant loader fit ~17B params (9B transformer + 8B Qwen3)
at 4-bit on 18 GB, and at what resident peak? (2) can it then generate (with tiled VAE)?

9B is the SAME architecture as 4B, just bigger config -- our modules parameterize directly,
the diffusers remaps + streaming loader apply unchanged.

    PYTHONPATH=. .venv/bin/python experiments/feasibility_9b.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image

from mxdiffusers.flux.latents import prepare_packed_latents, prepare_text_ids
from mxalloy.loader import QuantConfig, component_files, load_quantized
from mxdiffusers.flux.scheduler import FlowMatchEulerScheduler
from mxdiffusers.flux.text_encoder import Qwen3TextEncoder
from mxdiffusers.flux.tokenizer import KleinTokenizer
from mxdiffusers.flux.transformer import Flux2Transformer
from mxdiffusers.flux.vae import Flux2VAE
from mxdiffusers.flux.weight_mapping import (
    remap_text_encoder_key,
    remap_transformer_key,
    remap_vae_decode_key,
)

REPO = "black-forest-labs/FLUX.2-klein-9B"
PROMPT = "a brushed alloy sculpture under studio light"
SEED = 42
OUT_LAYERS = (9, 18, 27)


def gb(n: int) -> float:
    return round(n / 1e9, 2)


def main() -> None:
    model_dir = snapshot_download(
        REPO, local_files_only=True, ignore_patterns=["flux-2-klein-9b.safetensors"]
    )
    print(f"model dir: {model_dir}")

    # --- 9B config (same code as 4B, bigger numbers) ---
    mx.reset_peak_memory()
    transformer = Flux2Transformer(
        num_layers=8, num_single_layers=24, num_attention_heads=32, joint_attention_dim=12288
    )
    text_encoder = Qwen3TextEncoder(hidden_size=4096, intermediate_size=12288)
    vae = Flux2VAE()

    t0 = time.time()
    miss_t = load_quantized(
        transformer, component_files(model_dir, "transformer"),
        remap=remap_transformer_key, quant=QuantConfig(bits=4),
    )
    print(f"transformer loaded, peak {gb(mx.get_peak_memory())} GB, missing={len(miss_t)}")
    miss_e = load_quantized(
        text_encoder, component_files(model_dir, "text_encoder"),
        remap=remap_text_encoder_key, quant=QuantConfig(bits=4),
    )
    print(f"+text_encoder,  peak {gb(mx.get_peak_memory())} GB, missing={len(miss_e)}")
    miss_v = load_quantized(
        vae, component_files(model_dir, "vae"), remap=remap_vae_decode_key, quant=QuantConfig(bits=None)
    )
    resident = gb(mx.get_peak_memory())
    print(f"RESIDENT load peak: {resident} GB  ({time.time()-t0:.0f}s)  vae missing={len(miss_v)}")
    if miss_t or miss_e or miss_v:
        print(f"  WARNING missing params -> arch/remap mismatch: t={miss_t or ''} v={miss_v or ''}")

    tokenizer = KleinTokenizer(f"{model_dir}/tokenizer")

    def generate(h, w, steps=4, tile=64):
        mx.clear_cache(); mx.reset_peak_memory(); t = time.time()
        ids, am = tokenizer.encode(PROMPT)
        pe = text_encoder.get_prompt_embeds(ids, am, OUT_LAYERS)
        tids = prepare_text_ids(pe)
        lat, lids, lh, lw = prepare_packed_latents(seed=SEED, height=h, width=w, batch_size=1)
        sched = FlowMatchEulerScheduler(num_inference_steps=steps, image_seq_len=(h // 16) * (w // 16))
        for t_ in range(steps):
            n = transformer(hidden_states=lat, encoder_hidden_states=pe,
                            timestep=sched.timesteps[t_], img_ids=lids, txt_ids=tids, guidance=None)
            lat = sched.step(n, t_, lat); mx.eval(lat)
        packed = lat.reshape(1, lh, lw, lat.shape[-1]).transpose(0, 3, 1, 2)
        dec = vae.decode_packed_latents(packed, tile_latent=tile); mx.eval(dec)
        x = mx.clip(dec / 2 + 0.5, 0, 1)
        arr = (np.array(mx.transpose(x, (0, 2, 3, 1)).astype(mx.float32)) * 255).round().astype(np.uint8)
        Image.fromarray(arr[0]).save(f"experiments/klein9b_{w}x{h}.png")
        print(f"GEN {w}x{h} {steps} steps: {time.time()-t:.0f}s  peak {gb(mx.get_peak_memory())} GB")

    for (w, h) in [(512, 512), (1024, 1024)]:
        try:
            generate(h, w)
        except Exception as exc:  # noqa: BLE001
            print(f"GEN {w}x{h}: FAILED {type(exc).__name__}: {str(exc)[:140]}")


if __name__ == "__main__":
    main()
