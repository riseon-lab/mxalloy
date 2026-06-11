"""Flux2KleinEngine: end-to-end FLUX.2-klein text-to-image, resident (native MLX).

Builds the MMDiT transformer, Qwen3 text encoder, and VAE decoder, stream-loads the klein
checkpoint via mxalloy (transformer + encoder quantized; VAE bf16), and keeps everything
resident so repeated generations are warm. ``generate()`` mirrors the diffusers reference
pipeline: chat-template tokenize -> 3-layer stacked Qwen3 embeddings -> empirically-shifted
flow-match Euler denoise -> BN-denormalize -> unpatchify -> (optionally tiled) VAE decode.

The prompt's context projection inputs and the joint RoPE tables are computed once per
generation (they are step-constant). Classifier-free guidance runs only when ``guidance > 1``
(two transformer passes); the distilled klein default is 1.0 (off), matching the reference.

INTERNAL: requires mlx + transformers.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mxalloy.loader import QuantConfig, component_files, load_quantized
from mxdiffusers.flux.latents import image_ids, pack, patchify, text_ids, unpack, unpatchify
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

_TEXT_ENCODER_OUT_LAYERS = (9, 18, 27)  # diffusers klein reference: hidden_states[9/18/27]


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
        # Param paths each load left unpopulated — a non-empty set means a remap regression.
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

    def _encode(self, prompt: str) -> mx.array:
        ids, mask = self.tokenizer.encode(prompt)
        return self.text_encoder.get_prompt_embeds(ids, mask, _TEXT_ENCODER_OUT_LAYERS)

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
        context = self._encode(prompt).astype(mx.bfloat16)
        cfg = guidance > 1.0
        if cfg:
            negative = self._encode("").astype(mx.bfloat16)

        lh, lw = height // 8, width // 8  # 32-ch latent grid; packed grid is half again
        latents = mx.random.normal(
            (1, 32, lh, lw), key=mx.random.key(seed), dtype=mx.float32
        )
        packed = pack(patchify(latents))  # (1, lh/2*lw/2, 128)
        img_pos = image_ids(lh // 2, lw // 2)
        txt_pos = text_ids(context.shape[1])
        rope = Flux2Transformer.compute_rope(txt_pos, img_pos)  # step-constant
        sigmas = FlowMatchEulerScheduler.sigmas(steps, packed.shape[1])
        mx.eval(packed, context, rope[0], rope[1])

        for i in range(steps):
            sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
            t = mx.full((1,), sigma, dtype=mx.float32)
            x = packed.astype(mx.bfloat16)
            velocity = self.transformer(x, t, context, txt_pos, img_pos, rope=rope)
            if cfg:
                neg_velocity = self.transformer(x, t, negative, txt_pos, img_pos, rope=rope)
                velocity = neg_velocity + guidance * (velocity - neg_velocity)
            packed = FlowMatchEulerScheduler.step(
                packed, velocity.astype(mx.float32), sigma, sigma_next
            )
            mx.eval(packed)
            if on_step is not None:
                on_step(i + 1, steps)

        grid = unpack(packed, lh // 2, lw // 2)  # (1, 128, lh/2, lw/2)
        grid = self.vae.bn_denormalize_packed(grid)
        z = unpatchify(grid).astype(mx.bfloat16)  # (1, 32, lh, lw)
        decoded = self.vae.decode(z, tile_latent=self.vae_tile_latent)
        mx.eval(decoded)
        return self._to_pil(decoded)

    def set_loras(self, loras: list[tuple[str, float]]) -> dict:
        """Hot-swap active LoRAs on the resident transformer (pass ``[]`` to clear)."""
        from mxdiffusers.flux.lora import apply_loras, load_lora_file

        return apply_loras(
            self.transformer, [(load_lora_file(path), float(strength)) for path, strength in loras]
        )

    def clear_loras(self) -> None:
        """Remove all active LoRAs from the resident transformer."""
        from mxdiffusers.flux.lora import clear_loras

        clear_loras(self.transformer)

    @staticmethod
    def _to_pil(decoded: mx.array) -> Image.Image:
        x = mx.clip(decoded / 2 + 0.5, 0, 1)  # (B, H, W, 3) NHWC
        arr = np.array(x[0].astype(mx.float32)) * 255
        return Image.fromarray(arr.round().astype("uint8"))
