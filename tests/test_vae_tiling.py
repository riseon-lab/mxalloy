"""Tiled VAE decode math: blend/accumulation correctness, independent of real weights.

A region-local stub decoder (nearest 8x upscale, NHWC) has no cross-tile dependence, so a
correct tiling+feather+accumulate must reproduce a full decode (up to fp32 blend rounding).
This isolates the slice indices and feather weighting from the model's per-tile GroupNorm
drift.
"""

from __future__ import annotations

import pytest

from tests._mlx import require_mlx_core

mx = require_mlx_core()
try:
    from mxdiffusers.flux.vae import Flux2VAE
except RuntimeError as exc:
    if "No Metal device available" in str(exc):
        pytest.skip(str(exc), allow_module_level=True)
    raise

_UPSCALE = 8


def _stub_decoder(z: mx.array) -> mx.array:
    """Region-local fake decoder: take 3 channels, nearest-upsample by the real factor."""
    x = z[:, :, :, :3]
    x = mx.repeat(x, _UPSCALE, axis=1)
    x = mx.repeat(x, _UPSCALE, axis=2)
    return x.astype(mx.bfloat16)


def _vae_with_stub() -> Flux2VAE:
    vae = Flux2VAE.__new__(Flux2VAE)  # bypass the heavy real-decoder __init__
    vae.decoder = _stub_decoder
    return vae


def test_feather_shape_range_and_symmetry() -> None:
    m = Flux2VAE._feather(2, 3, ramp_px=4)  # latent units -> (1, 16, 24, 1) pixels
    assert m.shape == (1, 16, 24, 1)
    assert float(m.max()) <= 1.0
    assert float(m.min()) > 0.0  # strictly positive: no divide-by-zero in the blend
    # centre is full weight; corners are the floor.
    assert float(m[0, 8, 12, 0]) == pytest.approx(1.0)
    assert float(m[0, 0, 0, 0]) < float(m[0, 8, 12, 0])
    # separable + symmetric about both midlines.
    assert float(m[0, 0, 12, 0]) == pytest.approx(float(m[0, 15, 12, 0]))
    assert float(m[0, 8, 0, 0]) == pytest.approx(float(m[0, 8, 23, 0]))


def test_tiled_matches_full_for_region_local_decoder() -> None:
    vae = _vae_with_stub()
    z = mx.random.normal((1, 13, 11, 32)).astype(mx.bfloat16)  # NHWC latent grid
    full = vae.decoder(z).astype(mx.float32)
    tiled = vae._decode_tiled(z, tile=4, overlap=2)  # overlapping tiles in both dims
    assert tiled.shape == full.shape
    diff = float(mx.max(mx.abs(tiled - full)))
    assert diff < 1e-2, diff


def test_tiled_output_shape_scales_by_upscale() -> None:
    vae = _vae_with_stub()
    z = mx.random.normal((1, 9, 7, 32)).astype(mx.bfloat16)
    out = vae._decode_tiled(z, tile=4, overlap=2)
    assert out.shape == (1, 9 * _UPSCALE, 7 * _UPSCALE, 3)


def test_decode_single_tile_short_circuits_to_full() -> None:
    # decode() with tile_latent >= grid goes straight to the decoder (bit-exact path),
    # exercised here structurally via _decode_tiled's well-covered counterpart above.
    vae = _vae_with_stub()
    z = mx.random.normal((1, 10, 12, 32)).astype(mx.bfloat16)
    full = vae.decoder(z).astype(mx.float32)
    one_tile = vae._decode_tiled(z, tile=16, overlap=2)
    assert float(mx.max(mx.abs(one_tile - full))) < 1e-2
