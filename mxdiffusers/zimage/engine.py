"""Z-Image-Turbo engine: text -> image on the mxalloy runtime (native MLX).

Loads the S3-DiT transformer + Qwen3 text encoder (both quantized) + the AutoencoderKL VAE
(bf16) and keeps them resident. Generation mirrors the diffusers ZImagePipeline: encode with
``hidden_states[-2]`` (no pooling), few-step flow-match denoise with a static shift, the
model-timestep flip ``1-sigma`` and output negation, then ``latents/scaling + shift`` -> decode.

INTERNAL: requires mlx + transformers.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import AutoTokenizer

from mxalloy.loader import QuantConfig, component_files, load_quantized
from mxdiffusers.flux.text_encoder import Qwen3TextEncoder
from mxdiffusers.hub import resolve_model_dir
from mxdiffusers.zimage.transformer import ZImageTransformer
from mxdiffusers.zimage.vae import ZImageVAE
from mxdiffusers.zimage.weight_mapping import (
    remap_zimage_text_encoder_key,
    remap_zimage_transformer_key,
    remap_zimage_vae_key,
)

_HIDDEN_LAYER = -2  # diffusers ZImagePipeline uses text_encoder hidden_states[-2]
_STATIC_SHIFT = 3.0


_ZIMAGE_REPO = "Tongyi-MAI/Z-Image-Turbo"


def find_zimage_model_dir() -> str:
    return resolve_model_dir(None, default_repo=_ZIMAGE_REPO)


class ZImageEngine:
    def __init__(self, model_dir: str | None = None, quantize_bits: int | None = 4):
        model_dir = resolve_model_dir(model_dir, default_repo=_ZIMAGE_REPO)
        self.transformer = ZImageTransformer()
        self.text_encoder = Qwen3TextEncoder()
        self.vae = ZImageVAE()
        quant = QuantConfig(bits=quantize_bits)
        self.missing = {
            "transformer": load_quantized(
                self.transformer, component_files(model_dir, "transformer"),
                remap=remap_zimage_transformer_key, quant=quant,
            ),
            "text_encoder": load_quantized(
                self.text_encoder, component_files(model_dir, "text_encoder"),
                remap=remap_zimage_text_encoder_key, quant=quant,
            ),
            "vae": load_quantized(
                self.vae, component_files(model_dir, "vae"),
                remap=remap_zimage_vae_key, quant=QuantConfig(bits=None),
            ),
        }
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(Path(model_dir) / "tokenizer"), local_files_only=True
        )

    def _encode(self, prompt: str) -> mx.array:
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        toks = self.tokenizer(formatted, truncation=True, max_length=512, return_tensors="np")
        ids = mx.array(toks["input_ids"])
        # No padding for a single prompt -> all tokens are real (causal attn makes this exact),
        # so hidden_states[-2] needs no mask selection.
        _, hidden = self.text_encoder(
            ids, attention_mask=mx.ones(ids.shape, dtype=mx.int32), output_hidden_states=True
        )
        return hidden[_HIDDEN_LAYER][0]  # (cap_len, 2560)

    @staticmethod
    def _sigmas(steps: int) -> mx.array:
        s = mx.linspace(1.0, 1.0 / steps, steps)
        s = _STATIC_SHIFT * s / (1.0 + (_STATIC_SHIFT - 1.0) * s)  # static flow-match shift
        return mx.concatenate([s, mx.zeros((1,))])

    def generate(
        self,
        prompt: str,
        seed: int = 0,
        steps: int = 8,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 0.0,
        cache_threshold: float = 0.25,  # FBC is near-lossless on Z-Image -> on by default
        on_step: Callable[[int, int], None] | None = None,
    ) -> Image.Image:
        cap = self._encode(prompt).astype(mx.bfloat16)
        if hasattr(self.transformer, "reset_cache"):
            self.transformer.reset_cache()
        self.transformer.cache_threshold = cache_threshold
        lh, lw = height // 8, width // 8
        mx.random.seed(seed)
        latents = mx.random.normal((1, 16, lh, lw)).astype(mx.float32)
        sig = self._sigmas(steps)
        for i in range(steps):
            model_t = mx.array([1.0 - sig[i]])  # diffusers flips the timestep
            out = self.transformer(latents.astype(mx.bfloat16), model_t, cap)  # (16, lh, lw)
            noise_pred = -out[None].astype(mx.float32)  # diffusers negates the model output
            latents = latents + (sig[i + 1] - sig[i]) * noise_pred
            mx.eval(latents)
            if on_step is not None:
                on_step(i + 1, steps)
        decoded = self.vae.decode(latents.astype(mx.bfloat16))
        mx.eval(decoded)
        return self._to_pil(decoded)

    def set_loras(self, loras: list[tuple[str, float]]) -> dict:
        """Hot-swap active LoRAs on the resident Z-Image transformer."""
        from mxdiffusers.zimage.lora import apply_loras, load_lora_file

        return apply_loras(
            self.transformer, [(load_lora_file(path), float(strength)) for path, strength in loras]
        )

    def clear_loras(self) -> None:
        """Remove all active LoRAs from the resident Z-Image transformer."""
        from mxdiffusers.zimage.lora import clear_loras

        clear_loras(self.transformer)

    @staticmethod
    def _to_pil(decoded: mx.array) -> Image.Image:
        x = mx.clip(decoded / 2 + 0.5, 0, 1)
        x = mx.transpose(x, (0, 2, 3, 1)).astype(mx.float32)
        arr = (np.array(x) * 255).round().astype(np.uint8)
        return Image.fromarray(arr[0])
