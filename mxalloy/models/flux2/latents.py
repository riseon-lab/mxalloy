"""FLUX.2-klein latent preparation + packing (native MLX, no mflux dependency).

Initial-noise latents, grid position ids (image), token ids (text), and the pack/unpack
between spatial [B, C, H, W] and packed [B, H*W, C] layouts. Pure math.

INTERNAL: not part of the public API; requires mlx.
"""

from __future__ import annotations

import mlx.core as mx

PRECISION = mx.bfloat16


def prepare_grid_ids(latents: mx.array, t_coord: int) -> mx.array:
    batch_size, _, height, width = latents.shape
    h_ids = mx.arange(height, dtype=mx.int32)
    w_ids = mx.arange(width, dtype=mx.int32)
    h_grid = mx.broadcast_to(mx.expand_dims(h_ids, axis=1), (height, width))
    w_grid = mx.broadcast_to(mx.expand_dims(w_ids, axis=0), (height, width))
    flat_h = h_grid.reshape(-1)
    flat_w = w_grid.reshape(-1)
    t = mx.full(flat_h.shape, t_coord, dtype=mx.int32)
    layer_ids = mx.zeros_like(flat_h)
    coords = mx.stack([t, flat_h, flat_w, layer_ids], axis=1)
    coords = mx.expand_dims(coords, axis=0)
    return mx.broadcast_to(coords, (batch_size, coords.shape[1], coords.shape[2]))


def pack_latents(latents: mx.array) -> mx.array:
    batch_size, num_channels, height, width = latents.shape
    return latents.reshape(batch_size, num_channels, height * width).transpose(0, 2, 1)


def prepare_latents(
    seed: int,
    height: int,
    width: int,
    batch_size: int,
    num_latents_channels: int = 32,
    vae_scale_factor: int = 8,
) -> tuple[mx.array, mx.array, int, int]:
    height = 2 * (height // (vae_scale_factor * 2))
    width = 2 * (width // (vae_scale_factor * 2))
    latent_height = height // 2
    latent_width = width // 2
    latents = mx.random.normal(
        shape=(batch_size, num_latents_channels * 4, latent_height, latent_width),
        key=mx.random.key(seed),
    ).astype(PRECISION)
    latent_ids = prepare_grid_ids(latents, t_coord=0)
    return latents, latent_ids, latent_height, latent_width


def prepare_packed_latents(
    seed: int,
    height: int,
    width: int,
    batch_size: int,
    num_latents_channels: int = 32,
    vae_scale_factor: int = 8,
) -> tuple[mx.array, mx.array, int, int]:
    latents, latent_ids, latent_height, latent_width = prepare_latents(
        seed=seed,
        height=height,
        width=width,
        batch_size=batch_size,
        num_latents_channels=num_latents_channels,
        vae_scale_factor=vae_scale_factor,
    )
    return pack_latents(latents), latent_ids, latent_height, latent_width


def prepare_text_ids(x: mx.array, t_coord: mx.array | None = None) -> mx.array:
    batch_size, seq_len, _ = x.shape
    out_ids = []
    for i in range(batch_size):
        if t_coord is None:
            t = mx.zeros((seq_len,), dtype=mx.int32)
        else:
            t = t_coord[i]
            if t.ndim == 0:
                t = mx.full((seq_len,), t, dtype=mx.int32)
            elif t.shape[0] != seq_len:
                t = mx.broadcast_to(t, (seq_len,))
            t = t.astype(mx.int32)
        h = mx.zeros((seq_len,), dtype=mx.int32)
        w = mx.zeros((seq_len,), dtype=mx.int32)
        token_ids = mx.arange(seq_len, dtype=mx.int32)
        coords = mx.stack([t, h, w, token_ids], axis=1)
        out_ids.append(coords)
    return mx.stack(out_ids, axis=0)
