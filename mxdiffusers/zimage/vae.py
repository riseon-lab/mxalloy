"""Z-Image VAE decoder — stock diffusers ``AutoencoderKL`` (flux-dev, 16 latent ch).

Reuses the FLUX.2 decoder blocks (identical AutoencoderKL topology: conv_in -> mid block ->
4 up-decoder blocks -> group-norm -> conv_out) with a 16-channel ``conv_in`` and none of klein's
quant/post-quant conv, batch-norm, or 2x2 patch packing. INTERNAL; requires mlx.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mxdiffusers.flux.vae import Flux2Decoder


class ZImageVAE(nn.Module):
    """Decode-only Z-Image VAE (latents -> image)."""

    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159
    latent_channels: int = 16

    def __init__(self) -> None:
        super().__init__()
        self.decoder = Flux2Decoder(in_channels=self.latent_channels)

    def decode(self, latents: mx.array) -> mx.array:
        # latents (B, 16, H, W); diffusers applies scaling/shift before the decoder.
        latents = (latents / self.scaling_factor) + self.shift_factor
        return self.decoder(latents)
