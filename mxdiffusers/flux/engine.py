"""Flux2KleinEngine: end-to-end klein text-to-image, resident (native MLX).

Builds the transformer, Qwen3 text encoder, and VAE decoder, stream-loads the real klein
checkpoint into them (transformer + encoder quantized; VAE kept bf16), and keeps them
resident so repeated generations are warm. generate() runs tokenize -> encode -> 4-step
flow-match denoise -> decode -> image.

INTERNAL: requires mlx + transformers.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mxalloy.loader import QuantConfig, component_files, load_quantized
from mxdiffusers.flux.latents import prepare_packed_latents, prepare_text_ids
from mxdiffusers.flux.loader import find_klein_model_dir
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

_TEXT_ENCODER_OUT_LAYERS = (9, 18, 27)


class Flux2KleinEngine:
    def __init__(
        self,
        model_dir: str | None = None,
        quantize_bits: int | None = 4,
        vae_tile_latent: int | None = 128,
    ):
        # vae_tile_latent caps VAE-decode activations to a (tile*8)px region: 128 keeps
        # <=1024**2 a single (bit-exact) tile while letting larger images tile to fit 18 GB.
        # None disables tiling (full decode). Override upward on roomier machines.
        self.vae_tile_latent = vae_tile_latent
        model_dir = find_klein_model_dir(model_dir)
        self.transformer = Flux2Transformer()
        self.text_encoder = Qwen3TextEncoder()
        self.vae = Flux2VAE()
        quant = QuantConfig(bits=quantize_bits)
        # Param paths each load left unpopulated — a non-empty set means a remap regression
        # (unmatched checkpoint keys are dropped without trace; this is the integrity check).
        self.missing = {
            "transformer": load_quantized(
                self.transformer, component_files(model_dir, "transformer"),
                remap=remap_transformer_key, quant=quant,
            ),
            "text_encoder": load_quantized(
                self.text_encoder, component_files(model_dir, "text_encoder"),
                remap=remap_text_encoder_key, quant=quant,
            ),
            # VAE stays bf16 (decode is the memory peak; tiling handles it, not quantization).
            "vae": load_quantized(
                self.vae, component_files(model_dir, "vae"),
                remap=remap_vae_decode_key, quant=QuantConfig(bits=None),
            ),
        }
        self.tokenizer = KleinTokenizer(Path(model_dir) / "tokenizer")

    def generate(
        self,
        prompt: str,
        seed: int,
        steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 1.0,
        on_step: Callable[[int, int], None] | None = None,
    ) -> Image.Image:
        input_ids, attention_mask = self.tokenizer.encode(prompt)
        prompt_embeds = self.text_encoder.get_prompt_embeds(
            input_ids, attention_mask, _TEXT_ENCODER_OUT_LAYERS
        )
        text_ids = prepare_text_ids(prompt_embeds)

        # Reset the exact static-context cache for this generation (FLUX has no lossy cache).
        if hasattr(self.transformer, "reset_cache"):
            self.transformer.reset_cache()

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
                guidance=guidance,
            )
            latents = scheduler.step(noise, t, latents)
            mx.eval(latents)
            if on_step is not None:
                on_step(t + 1, steps)

        packed = latents.reshape(1, latent_height, latent_width, latents.shape[-1]).transpose(
            0, 3, 1, 2
        )
        decoded = self.vae.decode_packed_latents(packed, tile_latent=self.vae_tile_latent)
        mx.eval(decoded)
        return self._to_pil(decoded)

    def set_loras(self, loras: list[tuple[str, float]]) -> dict:
        """Hot-swap the active LoRA set on the resident base (no reload).

        ``loras`` = ``[(safetensors_path, strength), ...]``; pass ``[]`` to clear. Returns a
        ``{'applied': n, 'skipped': [...]}`` summary. Runtime-applied on the quantized weights.
        """
        from mxdiffusers.flux.lora import apply_loras, load_lora_file

        return apply_loras(
            self.transformer, [(load_lora_file(path), float(strength)) for path, strength in loras]
        )

    def clear_loras(self) -> None:
        """Remove all active LoRAs (restores the base bit-for-bit)."""
        from mxdiffusers.flux.lora import clear_loras

        clear_loras(self.transformer)

    @staticmethod
    def _to_pil(decoded: mx.array) -> Image.Image:
        x = mx.clip(decoded / 2 + 0.5, 0, 1)
        x = mx.transpose(x, (0, 2, 3, 1)).astype(mx.float32)
        arr = (np.array(x) * 255).round().astype(np.uint8)
        return Image.fromarray(arr[0])
