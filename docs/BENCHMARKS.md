# mxalloy Benchmarks — FLUX.2-klein-4B

Runtime and peak memory on Apple Silicon (**18 GB** unified memory), klein-4B quantized to
**4-bit** (transformer + Qwen3 text encoder; VAE kept bf16). The engine is **resident** and
reused across every config (warm path). Fixed prompt, seed 42. Peak = max active MLX
memory during generation (includes the ~4.5 GB resident weights). Reproduce with
`benchmarks/benchmark_klein.py`.

## Load
- Streaming-quantized load peak: **4.54 GB** (vs mflux's ~17.9 GB load-then-quantize), ~7 s.
- Model stays resident; every run below reused it — no per-image reload.

## Runtime + peak by resolution / steps

| Resolution | Aspect | MP | Steps | Time | Peak |
|---|---|---|---|---|---|
| 512×512 | 1:1 | 0.26 | 4 / 20 / 50 | 15.7 / 63.1 / 161.5 s | 7.46 GB |
| 896×512 | 16:9 | 0.46 | 4 / 20 | 21.8 / 97.8 s | 7.3 GB |
| 512×896 | 9:16 | 0.46 | 4 / 20 | 26.1 / 98.1 s | 7.3–8.4 GB |
| 1024×1024 | 1:1 | 1.05 | 4 / 20 / 50 | 52.4 / 207 / 500 s | 14.64 GB |
| 1344×768 | 16:9 | 1.03 | 4 / 20 | 54.2 / 195 s | 14.47 GB |
| 768×1344 | 9:16 | 1.03 | 4 / 20 | 51.9 / 193 s | 14.47 GB |
| 1080×1920 | 9:16 | 2.07 | — | skipped | ~24.5 GB (predicted) |
| 1920×1080 | 16:9 | 2.07 | — | skipped | ~24.5 GB (predicted) |
| 2048×2048 | 1:1 | 4.19 | — | skipped | ~44.9 GB (predicted) |

(Configs whose predicted peak exceeds `--budget-gb` are skipped to avoid thrashing swap.)

## Findings

- **Peak is pixel-count-bound and orientation-agnostic.** Square, 9:16, and 16:9 at the same
  megapixels peak within noise (~7.4 GB at ≤0.5 MP, ~14.5 GB at ~1 MP).
- **Peak is flat across step count** (it's per-forward activation, not cumulative), while
  **time is ~linear in steps** (~9.7 s/step at 1024²).
- The bottleneck above ~1 MP is **VAE-decode activations**, not weights — motivating tiled
  VAE decode as the next memory lever.
- On 18 GB, everything **≤ ~1 MP runs comfortably** (including portrait/landscape); ≥ 2 MP
  (HD, 2048²) needs more headroom: a larger machine or tiled VAE.

## Reproduce

```bash
# 18 GB: squares 512²/1024² across 4/20/50
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_klein.py --budget-gb 17

# aspect ratios
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_klein.py \
  --resolutions 896x512,512x896,1344x768,768x1344 --steps 4,20 --budget-gb 17

# larger machine (e.g. 48 GB): fill in HD + 2048²
PYTHONPATH=. .venv/bin/python benchmarks/benchmark_klein.py --budget-gb 44
```
