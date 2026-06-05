"""Tiled VAE decode math: blend/accumulation correctness, independent of real weights.

A region-local stub decoder (nearest 8x upscale) has no cross-tile dependence, so a
correct tiling+feather+accumulate must reproduce a full decode exactly. This isolates the
slice_update indices and feather weighting from the model's per-tile GroupNorm drift.
"""

from __future__ import annotations

import pytest

from tests._mlx import require_mlx_core

mx = require_mlx_core()
try:
    from mxdiffusers.flux.vae import _DECODER_UPSCALE, Flux2VAE, _feather
except RuntimeError as exc:
    if "No Metal device available" in str(exc):
        pytest.skip(str(exc), allow_module_level=True)
    raise


def _stub_decoder(z: mx.array) -> mx.array:
    """Region-local fake decoder: take 3 channels, nearest-upsample by the real factor."""
    x = z[:, :3, :, :]
    x = mx.repeat(x, _DECODER_UPSCALE, axis=2)
    x = mx.repeat(x, _DECODER_UPSCALE, axis=3)
    return x.astype(mx.bfloat16)


def _vae_with_stub() -> Flux2VAE:
    vae = Flux2VAE.__new__(Flux2VAE)  # bypass the heavy real-decoder __init__
    vae.decoder = _stub_decoder
    return vae


def test_feather_shape_range_and_symmetry() -> None:
    m = _feather(16, 24, fade=4)
    assert m.shape == (1, 1, 16, 24)
    arr = mx.array(m)
    assert float(arr.max()) <= 1.0
    assert float(arr.min()) > 0.0  # strictly positive: no divide-by-zero in the blend
    # centre is full weight; corners are the floor.
    assert float(arr[0, 0, 8, 12]) == pytest.approx(1.0)
    assert float(arr[0, 0, 0, 0]) < float(arr[0, 0, 8, 12])
    # separable + symmetric about both midlines.
    assert float(arr[0, 0, 0, 12]) == pytest.approx(float(arr[0, 0, 15, 12]))
    assert float(arr[0, 0, 8, 0]) == pytest.approx(float(arr[0, 0, 8, 23]))


def test_single_tile_is_exact_full_decode() -> None:
    vae = _vae_with_stub()
    z = mx.random.normal((1, 32, 10, 12)).astype(mx.bfloat16)
    full = vae.decoder(z)
    # tile >= both dims -> short-circuit straight to the decoder (bit-exact).
    tiled = vae._decode_tiled(z, tile_latent=16)
    assert mx.array_equal(full, tiled)


def test_tiled_matches_full_for_region_local_decoder() -> None:
    vae = _vae_with_stub()
    z = mx.random.normal((1, 32, 13, 11)).astype(mx.bfloat16)
    full = vae.decoder(z).astype(mx.float32)
    tiled = vae._decode_tiled(z, tile_latent=4)  # forces overlapping tiles in both dims
    assert tiled.shape == full.shape
    diff = float(mx.max(mx.abs(tiled - full)))
    assert diff < 1e-2, diff


def test_tiled_output_shape_scales_by_upscale() -> None:
    vae = _vae_with_stub()
    z = mx.random.normal((1, 32, 9, 7)).astype(mx.bfloat16)
    out = vae._decode_tiled(z, tile_latent=4)
    assert out.shape == (1, 3, 9 * _DECODER_UPSCALE, 7 * _DECODER_UPSCALE)
