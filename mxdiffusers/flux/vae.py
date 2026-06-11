"""FLUX.2 VAE (AutoencoderKLFlux2), decode path, native MLX.

Independent MLX reimplementation derived from the diffusers reference (Apache-2.0,
attributed in ``NOTICE``): a standard KL decoder over 32-channel latents with
``post_quant_conv``, plus klein's BatchNorm running stats (``bn.*``) which replace the usual
scaling/shift factors — the *engine* de-normalizes packed latents with them before
unpatchifying (mirroring the reference pipeline).

Adds feathered tiled decode (original mxalloy work): above ``tile_latent`` the latent grid is
decoded in overlapping tiles blended with a linear feather, holding decode peak memory flat
with resolution. A single tile is bit-exact full decode.

Interface: ``decode`` takes NCHW latents (the engine mirrors the reference pipeline's latent
math) and returns an NHWC image. INTERNAL: requires mlx.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mxdiffusers.vae_kl import VAEDecoder

_LATENT_CHANNELS = 32
_BN_EPS = 1e-4


class _BNStats(nn.Module):
    """Holds the checkpoint's packed-latent BatchNorm running stats (128 = 32ch x 2x2)."""

    def __init__(self) -> None:
        super().__init__()
        self.running_mean = mx.zeros((128,))
        self.running_var = mx.ones((128,))


class Flux2VAE(nn.Module):
    """Decode-only AutoencoderKLFlux2: ``post_quant_conv`` + ``decoder`` + ``bn`` stats."""

    batch_norm_eps = _BN_EPS

    def __init__(self) -> None:
        super().__init__()
        self.post_quant_conv = nn.Conv2d(_LATENT_CHANNELS, _LATENT_CHANNELS, 1)
        self.decoder = VAEDecoder(latent_channels=_LATENT_CHANNELS)
        self.bn = _BNStats()

    def bn_denormalize_packed(self, packed: mx.array) -> mx.array:
        """(B, 128, h/2, w/2) packed latents -> de-normalized via BN running stats."""
        mean = self.bn.running_mean.reshape(1, -1, 1, 1).astype(packed.dtype)
        std = mx.sqrt(self.bn.running_var + _BN_EPS).reshape(1, -1, 1, 1).astype(packed.dtype)
        return packed * std + mean

    def decode(self, latents: mx.array, tile_latent: int | None = None) -> mx.array:
        """(B, 32, h, w) NCHW latents -> (B, 8h, 8w, 3) NHWC image in [-1, 1]."""
        z = latents.transpose(0, 2, 3, 1)  # NHWC for the decoder stack
        z = self.post_quant_conv(z)
        b, h, w, _ = z.shape
        if tile_latent is None or (h <= tile_latent and w <= tile_latent):
            return self.decoder(z)  # single tile: bit-exact full decode
        return self._decode_tiled(z, tile_latent)

    def _decode_tiled(self, z: mx.array, tile: int, overlap: int = 16) -> mx.array:
        """Feathered overlapping tiles: peak memory ~ one tile's activations."""
        b, h, w, _ = z.shape
        out = mx.zeros((b, h * 8, w * 8, 3), dtype=mx.float32)
        weight = mx.zeros((1, h * 8, w * 8, 1), dtype=mx.float32)
        step = tile - overlap
        ys = list(range(0, max(h - overlap, 1), step))
        xs = list(range(0, max(w - overlap, 1), step))
        for y0 in ys:
            for x0 in xs:
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                piece = self.decoder(z[:, y0:y1, x0:x1, :]).astype(mx.float32)
                feather = self._feather(y1 - y0, x1 - x0, overlap * 8)
                out[:, y0 * 8 : y1 * 8, x0 * 8 : x1 * 8, :] = (
                    out[:, y0 * 8 : y1 * 8, x0 * 8 : x1 * 8, :] + piece * feather
                )
                weight[:, y0 * 8 : y1 * 8, x0 * 8 : x1 * 8, :] = (
                    weight[:, y0 * 8 : y1 * 8, x0 * 8 : x1 * 8, :] + feather
                )
                mx.eval(out, weight)  # free each tile's activations before the next
        return out / mx.maximum(weight, 1e-6)

    @staticmethod
    def _feather(h_lat: int, w_lat: int, ramp_px: int) -> mx.array:
        """(1, 8h, 8w, 1) weight: 1 in the interior, linear ramp toward tile edges."""
        hp, wp = h_lat * 8, w_lat * 8

        def ramp(n: int) -> mx.array:
            r = mx.minimum(mx.arange(n, dtype=mx.float32) + 1, float(ramp_px)) / float(ramp_px)
            return mx.minimum(r, r[::-1])

        return mx.minimum(ramp(hp)[:, None], ramp(wp)[None, :])[None, :, :, None]
