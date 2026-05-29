# mxalloy Benchmarks — FLUX.2-klein-4B

Runtime and peak memory on Apple Silicon (**18 GB** unified memory), klein-4B quantized to
**4-bit** (transformer + Qwen3 text encoder; VAE kept bf16). The engine is **resident** and
reused across every config (warm path). Fixed prompt, seed 42. Peak = max active MLX
memory during generation (includes the ~4.5 GB resident weights). **VAE decode is tiled by
default** (`vae_tile_latent=128`), which caps the decode peak — see below. Reproduce with
`benchmarks/benchmark_klein.py`.

## Load
- Streaming-quantized load peak: **4.54 GB** (vs mflux's ~17.9 GB load-then-quantize), ~7 s.
- Model stays resident; every run below reused it — no per-image reload.

## Runtime + peak by resolution / steps

Peak is the full-generation peak with default tiled decode. Tiling is a no-op below ~1 MP
(a single tile → **bit-exact** to full decode); above it the latent splits into overlapping
1024²-equivalent tiles, so the peak plateaus near one 1024² decode (~14.7 GB) instead of
scaling with pixels.

| Resolution | Aspect | MP | Steps | Time | Peak | Decode |
|---|---|---|---|---|---|---|
| 512×512 | 1:1 | 0.26 | 4 / 20 / 50 | 15.7 / 63.1 / 161.5 s | 7.46 GB | 1 tile (exact) |
| 896×512 | 16:9 | 0.46 | 4 / 20 | 21.8 / 97.8 s | 7.3 GB | 1 tile (exact) |
| 512×896 | 9:16 | 0.46 | 4 / 20 | 26.1 / 98.1 s | 7.3–8.4 GB | 1 tile (exact) |
| 1024×1024 | 1:1 | 1.05 | 4 / 20 / 50 | 52.4 / 207 / 500 s | 14.64 GB | 1 tile (exact) |
| 1344×768 | 16:9 | 1.03 | 4 | 58.8 s | 12.14 GB | 2 tiles |
| 768×1344 | 9:16 | 1.03 | 4 | 61.6 s | 12.14 GB | 2 tiles |
| 1920×1080 | 16:9 | 2.07 | 4 | 121.2 s | 14.68 GB | 6 tiles |
| 1080×1920 | 9:16 | 2.07 | 4 | 134.5 s | 14.68 GB | 6 tiles |
| 2048×2048 | 1:1 | 4.19 | 4 | 297.3 s | 14.72 GB | 9 tiles |

Peak is flat across step count, so the 4-step peak holds for any step count; time is ~linear
in steps (e.g. 2048² at 20 steps ≈ 5× the 4-step time). Resolutions round to a multiple of
16 (1080 → 1072). HD and 2048² previously **could not run** on 18 GB (full-decode peak
~24.5 GB and ~44.9 GB); tiling brings every config here under ~14.7 GB.

## Tiled VAE decode

Decode splits the latent into overlapping tiles (default `vae_tile_latent=128`, a 1024 px
tile after the 8× upsample), decodes one at a time (freeing each tile's activations before
the next), and blends them with a feathered mask. This caps decode — the phase that drove
the whole gen peak above ~1 MP — for a small per-tile normalization drift.

**Memory.** Decode peak stops scaling with pixels and plateaus at the single-tile cost:

| Resolution | Full decode | Tiled (128) |
|---|---|---|
| 1024² | 14.64 GB | 14.64 GB (1 tile) |
| 1344×768 | 14.47 GB | 12.14 GB |
| 1920×1080 | ~24.5 GB (OOM) | 14.68 GB |
| 2048² | ~44.9 GB (OOM) | 14.72 GB |

**Quality.** Tiles re-compute GroupNorm over their own region, so tiled ≠ full above one
tile. The drift shrinks with tile size (bigger tile → more stable stats):

| Config | Tile | Tiles | vs full decode |
|---|---|---|---|
| ≤ 1024² | 128 | 1 | **bit-exact** (diff 0) |
| 1344×768 | 128 | 2 | mean 0.84/255, max 13, no seams |
| 1024² | 64 | 9 | mean 9.7/255 — visible drift |

The default (128) keeps everything ≤ 1024² exact and large tiles stable, so realistic drift
is sub-1/255 with no seams — the feathered overlap hides tile edges (seam probe peaks at
1.9/255 at the boundary). Small tiles (e.g. 64 → ~7.5 GB ceiling) are only for extreme
memory pressure and trade visible quality for headroom.

## Findings

- **Tiled decode makes peak resolution-independent above ~1 MP** (~14.7 GB flat from 1024²
  to 2048²), bounded by one 1024²-equivalent tile rather than the full image. Everything
  through 2048² now fits 18 GB.
- **Peak is flat across step count** (per-forward activation, not cumulative), while **time
  is ~linear in steps** (~9.7 s/step at 1024²).
- Without tiling, peak is **pixel-count-bound and orientation-agnostic** (~7.4 GB ≤0.5 MP,
  ~14.5 GB ~1 MP) and the bottleneck above ~1 MP is **VAE-decode activations**, not weights
  — exactly what tiling caps.
- ≤ 1 MP is unaffected (single bit-exact tile); the win is purely at ≥ 2 MP, which tiling
  takes from out-of-reach to comfortable on 18 GB.

## Reproduce

```bash
# 18 GB: sweep at 4 steps, tiled decode on by default — nothing skips through 2048²
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_klein.py --steps 4 --budget-gb 17

# squares 512²/1024² across 4/20/50 (≤1 MP, unaffected by tiling)
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_klein.py \
  --resolutions 512x512,1024x1024 --steps 4,20,50 --budget-gb 17

# HD + 2048² generate with the default tiled engine (saves PNGs)
PYTHONPATH=. .venv/bin/python experiments/gen_tiled_hires.py

# tiled vs full decode: peak + pixel diff + seam probe at a given res/tile
PYTHONPATH=. .venv/bin/python experiments/verify_tiled_vae.py --width 1344 --height 768 --tile 128

# reproduce the pre-tiling (full-decode) peaks by disabling tiling
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_klein.py --vae-tile 0 --budget-gb 17

# a roomier machine (raise the budget): no tiling, fill in 20/50-step HD + 2048²
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_klein.py --vae-tile 0 --budget-gb 44
```
