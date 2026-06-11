"""FLUX.2 latent packing and position ids, native MLX.

Literal reimplementation of the reference pipeline's latent math (diffusers, Apache-2.0):
32-channel latents are 2x2-patchified channel-major to 128, flattened row-major to a token
sequence, and addressed by 4-axis (T, H, W, L) position ids — text tokens live on the L axis
at the origin, image tokens on (H, W) with T=L=0. mlx-free shapes, mlx ops; INTERNAL.
"""

from __future__ import annotations

import mlx.core as mx


def patchify(latents: mx.array) -> mx.array:
    """(B, C, H, W) -> (B, 4C, H/2, W/2), channel-major 2x2 patches."""
    b, c, h, w = latents.shape
    x = latents.reshape(b, c, h // 2, 2, w // 2, 2)
    x = x.transpose(0, 1, 3, 5, 2, 4)
    return x.reshape(b, c * 4, h // 2, w // 2)


def unpatchify(latents: mx.array) -> mx.array:
    """(B, 4C, H, W) -> (B, C, 2H, 2W) — inverse of :func:`patchify`."""
    b, c4, h, w = latents.shape
    c = c4 // 4
    x = latents.reshape(b, c, 2, 2, h, w)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(b, c, h * 2, w * 2)


def pack(latents: mx.array) -> mx.array:
    """(B, C, H, W) -> (B, H*W, C), row-major token sequence."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).transpose(0, 2, 1)


def unpack(tokens: mx.array, height: int, width: int) -> mx.array:
    """(B, H*W, C) -> (B, C, H, W) — inverse of :func:`pack` for row-major ids."""
    b, _, c = tokens.shape
    return tokens.transpose(0, 2, 1).reshape(b, c, height, width)


def image_ids(height: int, width: int) -> mx.array:
    """(H*W, 4) ids (0, h, w, 0) in row-major order."""
    hs = mx.repeat(mx.arange(height), width)
    ws = mx.tile(mx.arange(width), height)
    zeros = mx.zeros((height * width,), dtype=hs.dtype)
    return mx.stack([zeros, hs, ws, zeros], axis=1)


def text_ids(seq_len: int) -> mx.array:
    """(S, 4) ids (0, 0, 0, l)."""
    ls = mx.arange(seq_len)
    zeros = mx.zeros((seq_len,), dtype=ls.dtype)
    return mx.stack([zeros, zeros, zeros, ls], axis=1)
