"""Z-Image VAE decoder — stock diffusers ``AutoencoderKL`` (flux-dev family, 16 latent ch).

A 16-channel instance of the shared lineage-free decoder (``mxdiffusers.vae_kl``); the
flux-dev VAE has no quant/post-quant convs, and scaling/shift are applied here. The Z-Image
engine works NCHW (matching the diffusers reference math), so this wrapper transposes around
the NHWC decoder stack. INTERNAL; requires mlx.
"""

from __future__ import annotations

import mlx.core as mx
from mlx import nn

from mxdiffusers.vae_kl import VAEDecoder


class ZImageVAE(nn.Module):
    """Decode-only Z-Image VAE (latents -> image)."""

    scaling_factor: float = 0.3611
    shift_factor: float = 0.1159
    latent_channels: int = 16

    def __init__(self) -> None:
        super().__init__()
        self.decoder = VAEDecoder(latent_channels=self.latent_channels)

    def decode(self, latents: mx.array) -> mx.array:
        # latents (B, 16, H, W); diffusers applies scaling/shift before the decoder.
        z = (latents / self.scaling_factor) + self.shift_factor
        return self.decoder(z.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
