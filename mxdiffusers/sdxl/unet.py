"""SDXL UNet (UNet2DConditionModel), native MLX.

Independent MLX reimplementation derived from the diffusers reference (Apache-2.0).
Attribute names mirror the checkpoint state_dict exactly (``down_blocks.N.attentions.N.
transformer_blocks.N.attn1.to_q`` ...), so the weight remap is identity. NHWC layout; the
mxalloy loader transposes conv weights from the PyTorch checkpoint.

Config (stabilityai/stable-diffusion-xl-base-1.0, shared by Turbo + finetunes):
channels [320, 640, 1280]; down [Down, XAttnDown, XAttnDown]; up [XAttnUp, XAttnUp, Up];
2 resnets/block (3 on up); transformer layers/block [1, 2, 10]; 64-dim heads [5, 10, 20];
cross-attention dim 2048; text_time additional embeddings (pooled 1280 + six sinusoidal(256)
time_ids -> 2816 -> 1280).

INTERNAL: requires mlx.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

_BLOCK_OUT = (320, 640, 1280)
_T_LAYERS = (1, 2, 10)
_HEADS = (5, 10, 20)
_CROSS_DIM = 2048
_TIME_DIM = 1280
_GROUPS = 32


def timestep_embedding(t: mx.array, dim: int) -> mx.array:
    """Sinusoidal embedding, diffusers convention (flip_sin_to_cos=True, freq_shift=0)."""
    half = dim // 2
    exponent = -math.log(10000) * mx.arange(half, dtype=mx.float32) / half
    emb = t.astype(mx.float32)[:, None] * mx.exp(exponent)[None, :]
    return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_GROUPS, channels, eps=1e-5, affine=True, pytorch_compatible=True)


class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, dim: int = _TIME_DIM):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(x)))


class ResnetBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_emb_proj = nn.Linear(_TIME_DIM, out_ch)
        self.norm2 = _group_norm(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        if in_ch != out_ch:
            self.conv_shortcut = nn.Conv2d(in_ch, out_ch, 1)

    def __call__(self, x: mx.array, temb: mx.array) -> mx.array:
        h = self.conv1(nn.silu(self.norm1(x)))
        h = h + self.time_emb_proj(nn.silu(temb))[:, None, None, :]
        h = self.conv2(nn.silu(self.norm2(h)))
        if hasattr(self, "conv_shortcut"):
            x = self.conv_shortcut(x)
        return x + h


class CrossAttention(nn.Module):
    """attn1 (self, context=None) / attn2 (cross): 64-dim heads, no bias on q/k/v."""

    def __init__(self, dim: int, heads: int, context_dim: int | None = None):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        ctx = context_dim if context_dim is not None else dim
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(ctx, dim, bias=False)
        self.to_v = nn.Linear(ctx, dim, bias=False)
        self.to_out = [nn.Linear(dim, dim)]

    def __call__(self, x: mx.array, context: mx.array | None = None) -> mx.array:
        b, s, d = x.shape
        ctx = context if context is not None else x
        q = self.to_q(x).reshape(b, s, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.to_k(ctx).reshape(b, ctx.shape[1], self.heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.to_v(ctx).reshape(b, ctx.shape[1], self.heads, self.head_dim).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)
        return self.to_out[0](o.transpose(0, 2, 1, 3).reshape(b, s, d))


class GEGLUFeedForward(nn.Module):
    """ff.net.0.proj (dim -> 8*dim, gated) + ff.net.2 (4*dim -> dim); net.1 is Dropout."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = [_GEGLU(dim), _Identity(), nn.Linear(4 * dim, dim)]

    def __call__(self, x: mx.array) -> mx.array:
        return self.net[2](self.net[0](x))


class _GEGLU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 8 * dim)

    def __call__(self, x: mx.array) -> mx.array:
        x, gate = mx.split(self.proj(x), 2, axis=-1)
        return x * nn.gelu(gate)


class _Identity(nn.Module):  # placeholder so ff.net indices match the checkpoint (Dropout slot)
    def __call__(self, x: mx.array) -> mx.array:
        return x


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = CrossAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.attn2 = CrossAttention(dim, heads, context_dim=_CROSS_DIM)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = GEGLUFeedForward(dim)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        x = x + self.attn1(self.norm1(x))
        x = x + self.attn2(self.norm2(x), context)
        return x + self.ff(self.norm3(x))


class Transformer2D(nn.Module):
    """GN -> proj_in (linear) -> N BasicTransformerBlocks over (H*W) -> proj_out, residual."""

    def __init__(self, channels: int, heads: int, layers: int):
        super().__init__()
        # diffusers Transformer2DModel uses eps=1e-6 here (resnets/conv_norm_out use 1e-5)
        self.norm = nn.GroupNorm(_GROUPS, channels, eps=1e-6, affine=True, pytorch_compatible=True)
        self.proj_in = nn.Linear(channels, channels)
        self.transformer_blocks = [BasicTransformerBlock(channels, heads) for _ in range(layers)]
        self.proj_out = nn.Linear(channels, channels)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        b, h, w, c = x.shape
        residual = x
        y = self.proj_in(self.norm(x).reshape(b, h * w, c))
        for block in self.transformer_blocks:
            y = block(y, context)
        return self.proj_out(y).reshape(b, h, w, c) + residual


class Downsampler(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.conv(x)


class Upsampler(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        b, h, w, c = x.shape
        x = mx.broadcast_to(x[:, :, None, :, None, :], (b, h, 2, w, 2, c))
        return self.conv(x.reshape(b, h * 2, w * 2, c))


class DownBlock(nn.Module):
    """down_blocks.N: 2 resnets, optional attentions, optional downsampler."""

    def __init__(self, in_ch: int, out_ch: int, t_layers: int, heads: int, downsample: bool):
        super().__init__()
        self.resnets = [ResnetBlock(in_ch, out_ch), ResnetBlock(out_ch, out_ch)]
        if t_layers:
            self.attentions = [Transformer2D(out_ch, heads, t_layers) for _ in range(2)]
        if downsample:
            self.downsamplers = [Downsampler(out_ch)]

    def __call__(
        self, x: mx.array, temb: mx.array, context: mx.array
    ) -> tuple[mx.array, list[mx.array]]:
        residuals = []
        for i, resnet in enumerate(self.resnets):
            x = resnet(x, temb)
            if hasattr(self, "attentions"):
                x = self.attentions[i](x, context)
            residuals.append(x)
        if hasattr(self, "downsamplers"):
            x = self.downsamplers[0](x)
            residuals.append(x)
        return x, residuals


class MidBlock(nn.Module):
    def __init__(self, channels: int, t_layers: int, heads: int):
        super().__init__()
        self.resnets = [ResnetBlock(channels, channels), ResnetBlock(channels, channels)]
        self.attentions = [Transformer2D(channels, heads, t_layers)]

    def __call__(self, x: mx.array, temb: mx.array, context: mx.array) -> mx.array:
        x = self.resnets[0](x, temb)
        x = self.attentions[0](x, context)
        return self.resnets[1](x, temb)


class UpBlock(nn.Module):
    """up_blocks.N: 3 resnets consuming skips, optional attentions, optional upsampler."""

    def __init__(
        self,
        prev_ch: int,
        out_ch: int,
        skip_chs: tuple[int, int, int],
        t_layers: int,
        heads: int,
        upsample: bool,
    ):
        super().__init__()
        ins = [prev_ch + skip_chs[0], out_ch + skip_chs[1], out_ch + skip_chs[2]]
        self.resnets = [ResnetBlock(ins[i], out_ch) for i in range(3)]
        if t_layers:
            self.attentions = [Transformer2D(out_ch, heads, t_layers) for _ in range(3)]
        if upsample:
            self.upsamplers = [Upsampler(out_ch)]

    def __call__(
        self, x: mx.array, skips: list[mx.array], temb: mx.array, context: mx.array
    ) -> mx.array:
        for i, resnet in enumerate(self.resnets):
            x = mx.concatenate([x, skips.pop()], axis=-1)  # channel concat (NHWC)
            x = resnet(x, temb)
            if hasattr(self, "attentions"):
                x = self.attentions[i](x, context)
        if hasattr(self, "upsamplers"):
            x = self.upsamplers[0](x)
        return x


class SDXLUNet(nn.Module):
    """The SDXL denoiser: eps = unet(latents, t, prompt_embeds, pooled+time_ids embedding)."""

    def __init__(self) -> None:
        super().__init__()
        c0, c1, c2 = _BLOCK_OUT
        self.conv_in = nn.Conv2d(4, c0, 3, padding=1)
        self.time_embedding = TimestepEmbedding(c0)
        self.add_embedding = TimestepEmbedding(2816)
        self.down_blocks = [
            DownBlock(c0, c0, _T_LAYERS[0] * 0, _HEADS[0], downsample=True),  # plain DownBlock2D
            DownBlock(c0, c1, _T_LAYERS[1], _HEADS[1], downsample=True),
            DownBlock(c1, c2, _T_LAYERS[2], _HEADS[2], downsample=False),
        ]
        self.mid_block = MidBlock(c2, _T_LAYERS[2], _HEADS[2])
        self.up_blocks = [
            UpBlock(c2, c2, (c2, c2, c1), _T_LAYERS[2], _HEADS[2], upsample=True),
            UpBlock(c2, c1, (c1, c1, c0), _T_LAYERS[1], _HEADS[1], upsample=True),
            UpBlock(c1, c0, (c0, c0, c0), 0, _HEADS[0], upsample=False),
        ]
        self.conv_norm_out = _group_norm(c0)
        self.conv_out = nn.Conv2d(c0, 4, 3, padding=1)

    def __call__(
        self,
        latents: mx.array,  # (B, H/8, W/8, 4)
        timestep: mx.array,  # (B,)
        context: mx.array,  # (B, 77, 2048) concat CLIP-L | bigG penultimate
        pooled: mx.array,  # (B, 1280) bigG projected pooled
        time_ids: mx.array,  # (B, 6) [orig_h, orig_w, crop_t, crop_l, target_h, target_w]
    ) -> mx.array:
        dtype = latents.dtype
        temb = self.time_embedding(timestep_embedding(timestep, _BLOCK_OUT[0]).astype(dtype))
        ids_emb = timestep_embedding(time_ids.reshape(-1), 256).reshape(time_ids.shape[0], -1)
        aug = self.add_embedding(
            mx.concatenate([pooled, ids_emb.astype(dtype)], axis=-1)
        )
        temb = temb + aug

        x = self.conv_in(latents)
        skips = [x]
        for block in self.down_blocks:
            x, residuals = block(x, temb, context)
            skips.extend(residuals)
        x = self.mid_block(x, temb, context)
        for block in self.up_blocks:
            x = block(x, skips, temb, context)
        return self.conv_out(nn.silu(self.conv_norm_out(x)))
