# mxalloy

**The memory-lean inference runtime for Apple Silicon — plus `mxdiffusers`, a diffusers-style image stack built on it.**

`mxalloy` is the optimization layer beneath MLX: a streaming quantized model loader, device/runtime planning, tiled VAE decode, and attention primitives that let large models run on modest Apple Silicon with **dramatically lower peak memory** and no per-run reload. It ships no model code and is pure-Python.

`mxdiffusers` is the diffusion framework on top — a familiar `from_pretrained(...)` → `pipe(prompt)` API (**"diffusers for Mac"**) that delegates all the loading, quantization, and memory work to mxalloy. Two model families run today on an 18 GB Mac, through one `MXPipeline` base:

- **`MXFluxPipeline`** — FLUX.2-klein-4B (Black Forest Labs, Apache-2.0).
- **`MXZimagePipeline`** — Z-Image-Turbo-6B (Alibaba Tongyi, Apache-2.0), implemented **clean-room**.

## Quickstart

```bash
pip install "mxalloy[mlx]"                          # [mlx] is required to run a model
huggingface-cli download black-forest-labs/FLUX.2-klein-4B
```
```python
from mxdiffusers import MXFluxPipeline               # or: MXZimagePipeline

pipe = MXFluxPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B")  # 4-bit, resident
image = pipe("a brushed alloy sculpture, studio light", num_inference_steps=4).images[0]
image.save("out.png")
```
> Bare `pip install mxalloy` installs the mlx-free import surface (the loader/runtime API); the **`[mlx]` extra** pulls MLX and is required to actually load and run a model.

## Why mxalloy

Apple Silicon already runs diffusion via [mflux](https://github.com/filipstrand/mflux) (open + broad) and Draw Things (fast, closed). mxalloy is the **open, embeddable optimization layer** in between — model-agnostic by design:

- **Lowest peak memory, proven.** Streaming quantized load peaks at **4.5 GB** (4-bit) vs mflux's **17.9 GB** load-then-quantize on FLUX.2-klein-4B (**~3.9×**, same image). 8-bit fits 18 GB where mflux can't.
- **Faster end-to-end on constrained Macs.** ~**20% faster than mflux** at 512² and 1024² on 18 GB. The GEMMs are the *same* MLX kernels — the win is memory discipline: a smaller working set avoids the swap mflux falls into (its 1024² peak exceeds 18 GB).
- **Resolution decoupled from VRAM.** Tiled VAE keeps the FLUX generation peak **flat at ~14.7 GB from 1024² to 2048²** (≤1024² is bit-exact).
- **Two models, one API** — FLUX.2-klein and Z-Image-Turbo, both generating on 18 GB. Z-Image is a from-scratch MLX port against the diffusers reference (no mflux).
- **Opt-in step caching.** A first-block cache gives ~1.3× as an opt-in *fast mode*; the exact context/caption-projection cache is always on (output-neutral).
- **Resident + warm**, hot-swap LoRA (FLUX), embeddable on **stock MLX** — not a forked runtime.

## Measured (18 GB M3 Pro, 4-bit, warm)

| | mxalloy | mflux | |
|---|---|---|---|
| Load peak (klein-4B) | **4.5 GB** | 17.9 GB | ~3.9× lower |
| 512² gen (4-step klein) | **14.3 s / 7.5 GB** | 17.7 s / 12.5 GB | |
| 1024² gen (4-step klein) | **44.9 s / 14.6 GB** | 54.2 s / 19.7 GB* | *mflux swaps (>18 GB) |
| Tiled VAE peak, 1024²→2048² | **flat ~14.7 GB** | scales with pixels | HD/2048² on 18 GB |
| Z-Image-Turbo-6B | loads 6.2 GB, generates | — | clean-room MLX |

Per-GEMM compute is identical to mflux (same MLX); mxalloy's edge is memory. See `docs/BENCHMARKS.md`.

## Repository map

- **`mxalloy/`** — the runtime. `loader.py` (streaming quantized load: `load_quantized`, `QuantConfig`, `component_files`), `runtime/` (device + execution planning), `attention/` (pure-MLX fused quantized-KV attention — the live primitive), `kernels/`, `config.py` / `errors.py`.
- **`mxdiffusers/`** — diffusers-style pipelines on mxalloy: `pipeline.py` (`MXPipeline` base), `flux/` (`MXFluxPipeline`), `zimage/` (`MXZimagePipeline`). Consumes the runtime; the runtime never imports it (enforced by `tests/test_architecture_boundary.py`).
- **`surface/`** — a lean local Mac tester UI (model picker, LoRAs, refs, live memory). A test harness, not a product.
- **`research/`** — frozen experiments: an *experimental* compiled Metal `Primitive` for quantized-KV attention (correct but memory-not-speed on diffusion; **not built or shipped** — the pure-MLX path is what ships).
- **`benchmarks/`**, **`docs/`**, **`experiments/`** — repeatable benchmarks, design/versioning docs, research spikes.

## Honest status

- **Public API:** `mxalloy.load_quantized` / `QuantConfig` / `component_files`, the config dataclasses, and `mxalloy.errors`. The `mxdiffusers` pipeline API (`from_pretrained`/`__call__`) is stabilizing.
- **Attention:** the live primitive is pure-MLX. The compiled Metal kernel is frozen in `research/` — it's correct but, on the GEMM-bound diffusion path, memory-not-speed (it's for KV-cached/long-context workloads).
- **First-block cache** is opt-in (`cache_threshold`, default off), a *fidelity-for-speed* trade: near-lossless on Z-Image at 0.25 (visually identical), but on FLUX at 0.25 it shifts the result **visibly** — a *different* image of comparable quality, not a degraded one. Tune the threshold per model.

## License & provenance

Apache-2.0. `mxalloy` contains no model code and no mflux lineage. `mxdiffusers/flux` was ported from / verified against [mflux](https://github.com/filipstrand/mflux) (MIT) — see `mxdiffusers/PROVENANCE.md`; `mxdiffusers/zimage` is clean-room against the `diffusers` reference. Model weights are downloaded under their own licenses (both shipping models are Apache-2.0); mxalloy bundles none.
