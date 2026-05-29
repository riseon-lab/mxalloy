"""Flux2KleinEngine: end-to-end klein text-to-image, resident (native MLX).

Builds the transformer, Qwen3 text encoder, and VAE decoder, stream-loads the real klein
checkpoint into them (transformer + encoder quantized; VAE kept bf16), and keeps them
resident so repeated generations are warm. generate() runs tokenize -> encode -> 4-step
flow-match denoise -> decode -> image.

INTERNAL: requires mlx + transformers.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mxalloy.models.flux2.latents import prepare_packed_latents, prepare_text_ids
from mxalloy.models.flux2.loader import component_files, find_klein_model_dir, load_into_module
from mxalloy.models.flux2.scheduler import FlowMatchEulerScheduler
from mxalloy.models.flux2.text_encoder import Qwen3TextEncoder
from mxalloy.models.flux2.tokenizer import KleinTokenizer
from mxalloy.models.flux2.transformer import Flux2Transformer
from mxalloy.models.flux2.vae import Flux2VAE
from mxalloy.models.flux2.weight_mapping import (
    remap_text_encoder_key,
    remap_transformer_key,
    remap_vae_decode_key,
)

_TEXT_ENCODER_OUT_LAYERS = (9, 18, 27)


class Flux2KleinEngine:
    def __init__(self, model_dir: str | None = None, quantize_bits: int | None = 4):
        model_dir = model_dir or find_klein_model_dir()
        self.transformer = Flux2Transformer()
        self.text_encoder = Qwen3TextEncoder()
        self.vae = Flux2VAE()
        load_into_module(
            self.transformer,
            component_files(model_dir, "transformer"),
            remap_transformer_key,
            quantize_bits=quantize_bits,
        )
        load_into_module(
            self.text_encoder,
            component_files(model_dir, "text_encoder"),
            remap_text_encoder_key,
            quantize_bits=quantize_bits,
        )
        load_into_module(self.vae, component_files(model_dir, "vae"), remap_vae_decode_key)
        self.tokenizer = KleinTokenizer(Path(model_dir) / "tokenizer")

    def generate(
        self,
        prompt: str,
        seed: int,
        steps: int = 4,
        height: int = 1024,
        width: int = 1024,
    ) -> Image.Image:
        input_ids, attention_mask = self.tokenizer.encode(prompt)
        prompt_embeds = self.text_encoder.get_prompt_embeds(
            input_ids, attention_mask, _TEXT_ENCODER_OUT_LAYERS
        )
        text_ids = prepare_text_ids(prompt_embeds)

        latents, latent_ids, latent_height, latent_width = prepare_packed_latents(
            seed=seed, height=height, width=width, batch_size=1
        )
        image_seq_len = (height // 16) * (width // 16)
        scheduler = FlowMatchEulerScheduler(num_inference_steps=steps, image_seq_len=image_seq_len)

        for t in range(steps):
            noise = self.transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=scheduler.timesteps[t],
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=None,
            )
            latents = scheduler.step(noise, t, latents)
            mx.eval(latents)

        packed = latents.reshape(1, latent_height, latent_width, latents.shape[-1]).transpose(
            0, 3, 1, 2
        )
        decoded = self.vae.decode_packed_latents(packed)
        mx.eval(decoded)
        return self._to_pil(decoded)

    @staticmethod
    def _to_pil(decoded: mx.array) -> Image.Image:
        x = mx.clip(decoded / 2 + 0.5, 0, 1)
        x = mx.transpose(x, (0, 2, 3, 1)).astype(mx.float32)
        arr = (np.array(x) * 255).round().astype(np.uint8)
        return Image.fromarray(arr[0])
