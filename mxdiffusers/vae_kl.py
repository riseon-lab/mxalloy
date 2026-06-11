"""Shared AutoencoderKL decoder blocks (decode path), native MLX, NHWC.

Independent MLX reimplementation derived from the diffusers ``AutoencoderKL`` decoder
reference (Apache-2.0, attributed in ``NOTICE``): conv_in -> mid (resnet/attention/resnet) ->
up blocks (3 resnets each, nearest-2x upsample between) -> group-norm -> silu -> conv_out.
Attribute names mirror the diffusers state_dict (``decoder.*`` subtree) so family remaps are
identity. One parameterised decoder serves every KL family: SDXL (4ch), FLUX-dev/Z-Image
(16ch), FLUX.2 (32ch).

First verified as part of the SDXL family (same-latents image parity vs diffusers); promoted
here so all families share one lineage-free implementation. INTERNAL: requires mlx.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

_GROUPS = 32


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_GROUPS, channels, eps=1e-6, affine=True, pytorch_compatible=True)


def upsample_nearest_2x(x: mx.array) -> mx.array:
    b, h, w, c = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (b, h, 2, w, 2, c))
    return x.reshape(b, h * 2, w * 2, c)


class VAEResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        if in_ch != out_ch:
            self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        h = self.conv2(nn.silu(self.norm2(h)))
        if hasattr(self, "conv_shortcut"):
            x = self.conv_shortcut(x)
        return x + h


class VAEAttention(nn.Module):
    """Single-head spatial self-attention over (H*W) tokens (diffusers ``Attention``)."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.group_norm = _group_norm(channels)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = [nn.Linear(channels, channels)]

    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        y = self.group_norm(x).reshape(b, h * w, c)
        q = self.to_q(y)[:, None]
        k = self.to_k(y)[:, None]
        v = self.to_v(y)[:, None]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=c**-0.5)[:, 0]
        return x + self.to_out[0](y).reshape(b, h, w, c)


class VAEMidBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.resnets = [VAEResnetBlock(channels, channels), VAEResnetBlock(channels, channels)]
        self.attentions = [VAEAttention(channels)]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        return self.resnets[1](x)


class _Upsampler(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(upsample_nearest_2x(x))


class VAEUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, upsample: bool):
        super().__init__()
        self.resnets = [VAEResnetBlock(in_ch if i == 0 else out_ch, out_ch) for i in range(3)]
        if upsample:
            self.upsamplers = [_Upsampler(out_ch)]

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if hasattr(self, "upsamplers"):
            x = self.upsamplers[0](x)
        return x


class VAEDecoder(nn.Module):
    """``decoder.*`` tree: conv_in -> mid -> reversed-channel up blocks -> norm/act/conv_out."""

    def __init__(
        self,
        latent_channels: int,
        block_out: tuple[int, ...] = (128, 256, 512, 512),
        out_channels: int = 3,
    ):
        super().__init__()
        rev = list(reversed(block_out))  # e.g. [512, 512, 256, 128]
        self.conv_in = nn.Conv2d(latent_channels, rev[0], 3, padding=1)
        self.mid_block = VAEMidBlock(rev[0])
        self.up_blocks = [
            VAEUpBlock(rev[max(i - 1, 0)], rev[i], upsample=i < len(rev) - 1)
            for i in range(len(rev))
        ]
        self.conv_norm_out = _group_norm(rev[-1])
        self.conv_out = nn.Conv2d(rev[-1], out_channels, 3, padding=1)

    def __call__(self, z: mx.array) -> mx.array:
        """(B, H, W, latent_channels) -> (B, 8H, 8W, out_channels), NHWC."""
        x = self.conv_in(z)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(nn.silu(self.conv_norm_out(x)))
