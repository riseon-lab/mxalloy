"""SDXL VAE decoder (AutoencoderKL, decode path only), native MLX.

A 4-channel-latent instance of the shared lineage-free decoder (``mxdiffusers.vae_kl``),
plus SDXL's ``post_quant_conv`` and 0.13025 scaling. NHWC layout; the mxalloy loader
transposes conv weights.

Weights load from the fp32 shard and are cast to bf16 by the engine — fp16's *range* is what
NaNs the SDXL VAE; bf16 keeps fp32's exponent. INTERNAL: requires mlx.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mxdiffusers.vae_kl import VAEDecoder


class SDXLVAE(nn.Module):
    """Decode-only AutoencoderKL: ``post_quant_conv`` + ``decoder``. scaling 0.13025."""

    scaling_factor = 0.13025

    def __init__(self) -> None:
        super().__init__()
        self.post_quant_conv = nn.Conv2d(4, 4, 1)
        self.decoder = VAEDecoder(latent_channels=4)

    def decode(self, latents: mx.array) -> mx.array:
        """(B, H/8, W/8, 4) scaled latents -> (B, H, W, 3) image in [-1, 1]."""
        return self.decoder(self.post_quant_conv(latents / self.scaling_factor))
