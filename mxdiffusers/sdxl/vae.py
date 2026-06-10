"""SDXL VAE decoder (AutoencoderKL, decode path only), native MLX.

Independent MLX reimplementation derived from the diffusers ``AutoencoderKL`` reference
(Apache-2.0). Attribute names mirror the checkpoint state_dict (``decoder.*``,
``post_quant_conv``) so the remap is identity-plus-filter. NHWC layout (mlx-native); the
mxalloy loader transposes conv weights.

Weights load from the fp32 shard and are cast to bf16 by the engine — fp16's *range* is what
NaNs the SDXL VAE; bf16 keeps fp32's exponent. Decode peak at 1024² is modest for this 4ch
VAE; tiling is not needed at SDXL's native resolutions.

INTERNAL: requires mlx.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

_GROUPS = 32


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_GROUPS, channels, eps=1e-6, affine=True, pytorch_compatible=True)


def _upsample_nearest_2x(x: mx.array) -> mx.array:
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


class VAEUpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, upsample: bool):
        super().__init__()
        self.resnets = [
            VAEResnetBlock(in_ch if i == 0 else out_ch, out_ch) for i in range(3)
        ]
        if upsample:
            self.upsamplers = [_Upsampler(out_ch)]

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        if hasattr(self, "upsamplers"):
            x = self.upsamplers[0](x)
        return x


class _Upsampler(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(_upsample_nearest_2x(x))


class VAEDecoder(nn.Module):
    """decoder.* tree: conv_in -> mid -> up_blocks (512,512,256,128) -> norm/act/conv_out."""

    def __init__(self, latent_channels: int = 4, block_out: tuple[int, ...] = (128, 256, 512, 512)):
        super().__init__()
        rev = list(reversed(block_out))  # [512, 512, 256, 128]
        self.conv_in = nn.Conv2d(latent_channels, rev[0], 3, padding=1)
        self.mid_block = VAEMidBlock(rev[0])
        self.up_blocks = [
            VAEUpBlock(rev[max(i - 1, 0)], rev[i], upsample=i < len(rev) - 1)
            for i in range(len(rev))
        ]
        self.conv_norm_out = _group_norm(rev[-1])
        self.conv_out = nn.Conv2d(rev[-1], 3, 3, padding=1)

    def __call__(self, z: mx.array) -> mx.array:
        x = self.conv_in(z)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        return self.conv_out(nn.silu(self.conv_norm_out(x)))


class SDXLVAE(nn.Module):
    """Decode-only AutoencoderKL: ``post_quant_conv`` + ``decoder``. scaling 0.13025."""

    scaling_factor = 0.13025

    def __init__(self) -> None:
        super().__init__()
        self.post_quant_conv = nn.Conv2d(4, 4, 1)
        self.decoder = VAEDecoder()

    def decode(self, latents: mx.array) -> mx.array:
        """(B, H/8, W/8, 4) scaled latents -> (B, H, W, 3) image in [-1, 1]."""
        return self.decoder(self.post_quant_conv(latents / self.scaling_factor))
