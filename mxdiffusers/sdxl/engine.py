"""SDXL engine: text -> image on the mxalloy runtime (native MLX).

Loads the UNet + both CLIP text encoders (quantized 4-bit) and the VAE decoder (bf16,
upcast from the fp32 shard — fp16's range NaNs the SDXL VAE) and keeps them resident.
Generation follows the diffusers ``StableDiffusionXLPipeline`` reference: dual-CLIP
penultimate embeddings + bigG pooled, six micro-conditioning time_ids, Euler denoise with
classifier-free guidance (the cond/uncond pair batched), then ``latents/0.13025`` -> decode.

INTERNAL: requires mlx + transformers.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map
from PIL import Image
from transformers import CLIPTokenizer

from mxalloy.loader import QuantConfig, component_files, load_quantized
from mxdiffusers.hub import resolve_model_dir
from mxdiffusers.sdxl.clip import CLIP_BIGG, CLIP_L, CLIPTextEncoder
from mxdiffusers.sdxl.scheduler import EulerDiscreteScheduler
from mxdiffusers.sdxl.unet import SDXLUNet
from mxdiffusers.sdxl.vae import SDXLVAE
from mxdiffusers.sdxl.weight_mapping import (
    remap_sdxl_text_encoder_key,
    remap_sdxl_unet_key,
    remap_sdxl_vae_key,
)

_SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
_DTYPE = mx.float16  # checkpoint dtype; the VAE is upcast to bf16 below


def find_sdxl_model_dir() -> str:
    return resolve_model_dir(None, default_repo=_SDXL_REPO)


class SDXLEngine:
    def __init__(self, model_dir: str | None = None, quantize_bits: int | None = 4):
        model_dir = resolve_model_dir(model_dir, default_repo=_SDXL_REPO)
        self.unet = SDXLUNet()
        self.text_encoder = CLIPTextEncoder(CLIP_L)
        self.text_encoder_2 = CLIPTextEncoder(CLIP_BIGG)
        self.vae = SDXLVAE()
        quant = QuantConfig(bits=quantize_bits)
        self.missing = {
            "unet": load_quantized(
                self.unet, component_files(model_dir, "unet"),
                remap=remap_sdxl_unet_key, quant=quant,
            ),
            "text_encoder": load_quantized(
                self.text_encoder, component_files(model_dir, "text_encoder"),
                remap=remap_sdxl_text_encoder_key, quant=quant,
            ),
            "text_encoder_2": load_quantized(
                self.text_encoder_2, component_files(model_dir, "text_encoder_2"),
                remap=remap_sdxl_text_encoder_key, quant=quant,
            ),
            "vae": load_quantized(
                self.vae, component_files(model_dir, "vae"),
                remap=remap_sdxl_vae_key, quant=QuantConfig(bits=None),
            ),
        }
        # fp32 shard -> bf16 compute: keeps fp32's exponent range (fp16 NaNs the SDXL VAE).
        self.vae.update(tree_map(lambda a: a.astype(mx.bfloat16), self.vae.parameters()))
        self.tokenizer = CLIPTokenizer.from_pretrained(
            str(Path(model_dir) / "tokenizer"), local_files_only=True
        )
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(
            str(Path(model_dir) / "tokenizer_2"), local_files_only=True
        )
        self.scheduler = EulerDiscreteScheduler()

    def _encode_one(self, prompt: str) -> tuple[mx.array, mx.array]:
        """prompt -> (context (1, 77, 2048), pooled (1, 1280))."""
        ids1 = self.tokenizer(
            prompt, padding="max_length", max_length=77, truncation=True
        ).input_ids
        ids2 = self.tokenizer_2(
            prompt, padding="max_length", max_length=77, truncation=True
        ).input_ids
        h1, _ = self.text_encoder(mx.array([ids1]))
        h2, pooled = self.text_encoder_2(mx.array([ids2]))
        assert pooled is not None
        return mx.concatenate([h1, h2], axis=-1), pooled

    def generate(
        self,
        prompt: str,
        seed: int,
        steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 5.0,
        negative_prompt: str = "",
        on_step: Callable[[int, int], None] | None = None,
    ) -> Image.Image:
        context_c, pooled_c = self._encode_one(prompt)
        cfg = guidance > 1.0
        if cfg:
            context_u, pooled_u = self._encode_one(negative_prompt)
            context = mx.concatenate([context_u, context_c])
            pooled = mx.concatenate([pooled_u, pooled_c])
        else:
            context, pooled = context_c, pooled_c
        mx.eval(context, pooled)

        timesteps, sigmas = self.scheduler.make_schedule(steps)
        latents = (
            mx.random.normal(
                (1, height // 8, width // 8, 4), key=mx.random.key(seed), dtype=mx.float32
            )
            * self.scheduler.init_noise_sigma(sigmas)
        )
        time_ids = mx.array(
            [[height, width, 0, 0, height, width]] * context.shape[0], dtype=mx.float32
        )

        batch = context.shape[0]
        for i in range(steps):
            sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
            x = self.scheduler.scale_model_input(latents, sigma).astype(_DTYPE)
            x = mx.concatenate([x] * batch)
            t = mx.full((batch,), float(timesteps[i]), dtype=mx.float32)
            eps = self.unet(x, t, context, pooled, time_ids).astype(mx.float32)
            if cfg:
                eps_u, eps_c = eps[0:1], eps[1:2]
                eps = eps_u + guidance * (eps_c - eps_u)
            latents = self.scheduler.step(latents, eps, sigma, sigma_next)
            mx.eval(latents)
            if on_step is not None:
                on_step(i + 1, steps)

        decoded = self.vae.decode(latents.astype(mx.bfloat16))
        mx.eval(decoded)
        return self._to_pil(decoded)

    def set_loras(self, loras: list[tuple[str, float]]) -> dict:
        """Hot-swap active LoRAs on the resident SDXL UNet (pass ``[]`` to clear)."""
        from mxdiffusers.sdxl.lora import apply_loras, load_lora_file

        return apply_loras(
            self.unet, [(load_lora_file(path), float(strength)) for path, strength in loras]
        )

    def clear_loras(self) -> None:
        """Remove all active LoRAs from the resident SDXL UNet."""
        from mxdiffusers.sdxl.lora import clear_loras

        clear_loras(self.unet)

    @staticmethod
    def _to_pil(decoded: mx.array) -> Image.Image:
        x = mx.clip(decoded / 2 + 0.5, 0, 1)
        arr = np.array(x[0].astype(mx.float32)) * 255
        return Image.fromarray(arr.round().astype("uint8"))
