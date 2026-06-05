# mxalloy

<p align="center">
  <img src="assets/mxalloy-logo.png" alt="mxalloy metal logo" width="900">
</p>

**The memory-lean inference runtime for Apple Silicon — plus image and audio pipeline stacks built on it.**

`mxalloy` is the optimization layer beneath MLX: a streaming quantized model loader, device/runtime planning, tiled VAE decode, and attention primitives that let large models run on modest Apple Silicon with **dramatically lower peak memory** and no per-run reload. It ships no model code and is pure-Python.

`mxdiffusers` is the diffusion framework on top — a familiar `from_pretrained(...)` → `pipe(prompt)` API (**"diffusers for Mac"**) that delegates all the loading, quantization, and memory work to mxalloy. Two model families run today on an 18 GB Mac, through one `MXPipeline` base:

- **`MXFluxPipeline`** — FLUX.2-klein-4B (Black Forest Labs, Apache-2.0). Current implementation is a close MLX port verified against mflux, with MIT attribution carried in `NOTICE`.
- **`MXZimagePipeline`** — Z-Image-Turbo-6B (Alibaba Tongyi, Apache-2.0). The transformer is clean-room against the `diffusers` reference; the current package still reuses shared FLUX-derived Qwen/VAE helpers until those move to independent shared modules.

`mxtts` is the matching speech stack. Its first pipeline, **`MXMisoTTSPipeline`**, is a
hybrid adapter for Miso Labs' Miso TTS 8B: it gives Alloy a stable audio API while the native
MLX/quantized backend is mapped.

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

### Miso TTS Spike

```bash
git clone https://github.com/MisoLabsAI/MisoTTS.git ../MisoTTS
cd ../MisoTTS && uv sync --python 3.10 && cd -
../MisoTTS/.venv/bin/python examples/miso_text_to_speech.py \
  --source-path ../MisoTTS \
  --text "Hello from Alloy." \
  --output outputs/miso.wav
```

This uses the upstream PyTorch/Moshi runtime for now. Native mxalloy quantized load for the
8B generator is tracked in `docs/MISO_TTS_PLAN.md`.

## Why mxalloy

Apple Silicon already runs diffusion via [mflux](https://github.com/filipstrand/mflux) (open + broad) and Draw Things (fast, closed). mxalloy is the **open, embeddable optimization layer** in between — model-agnostic by design:

- **Lowest peak memory, proven.** Streaming quantized load peaks at **4.5 GB** (4-bit) vs mflux's **17.9 GB** load-then-quantize on FLUX.2-klein-4B (**~3.9×**, same image). 8-bit fits 18 GB where mflux can't.
- **Faster end-to-end on constrained Macs.** ~**20% faster than mflux** at 512² and 1024² on 18 GB. The GEMMs are the *same* MLX kernels — the win is memory discipline: a smaller working set avoids the swap mflux falls into (its 1024² peak exceeds 18 GB).
- **Resolution decoupled from VRAM.** Tiled VAE keeps the FLUX generation peak **flat at ~14.7 GB from 1024² to 2048²** (≤1024² is bit-exact).
- **Two models, one API** — FLUX.2-klein and Z-Image-Turbo, both generating on 18 GB. Z-Image's transformer is a from-scratch MLX port against the diffusers reference; shared Qwen/VAE helpers are called out in provenance.
- **Step caching, per model.** The exact context/caption-projection cache is always on (output-neutral, both models); a first-block cache (~1.3×) is **on by default for Z-Image** (near-lossless there) and **excluded from FLUX** (where it would visibly shift the image).
- **Resident + warm**, hot-swap LoRA (FLUX), embeddable on **stock MLX** — not a forked runtime.

## Measured (18 GB M3 Pro, 4-bit, warm)

| | mxalloy | mflux | |
|---|---|---|---|
| Load peak (klein-4B) | **4.5 GB** | 17.9 GB | ~3.9× lower |
| 512² gen (4-step klein) | **14.3 s / 7.5 GB** | 17.7 s / 12.5 GB | |
| 1024² gen (4-step klein) | **44.9 s / 14.6 GB** | 54.2 s / 19.7 GB* | *mflux swaps (>18 GB) |
| Tiled VAE peak, 1024²→2048² | **flat ~14.7 GB** | scales with pixels | HD/2048² on 18 GB |
| Z-Image-Turbo-6B | loads 6.2 GB, generates | — | clean-room transformer; shared helpers |

Per-GEMM compute is identical to mflux (same MLX); mxalloy's edge is memory. See `docs/BENCHMARKS.md`.

## Repository map

- **`mxalloy/`** — the runtime. `loader.py` (streaming quantized load: `load_quantized`, `QuantConfig`, `component_files`), `runtime/` (device + execution planning), `attention/` (pure-MLX fused quantized-KV attention — the live primitive), `kernels/`, `config.py` / `errors.py`.
- **`mxdiffusers/`** — diffusers-style pipelines on mxalloy: `pipeline.py` (`MXPipeline` base), `flux/` (`MXFluxPipeline`), `zimage/` (`MXZimagePipeline`). Consumes the runtime; the runtime never imports it (enforced by `tests/test_architecture_boundary.py`).
- **`mxtts/`** — text-to-speech/audio pipelines on mxalloy: `pipeline.py` (`MXTTSPipeline` base), `miso/` (`MXMisoTTSPipeline`). The current Miso path is a hybrid upstream adapter while the native MLX backend is being mapped.
- **`surface/`** — a lean local Mac tester UI (model picker, LoRAs, refs, live memory). A test harness, not a product.
- **`research/`** — frozen experiments: an *experimental* compiled Metal `Primitive` for quantized-KV attention (correct but memory-not-speed on diffusion; **not built or shipped** — the pure-MLX path is what ships).
- **`benchmarks/`**, **`docs/`**, **`experiments/`** — repeatable benchmarks, design/versioning docs, research spikes.

## Honest status

- **Public API:** `mxalloy.load_quantized` / `QuantConfig` / `component_files`, the config dataclasses, and `mxalloy.errors`. The `mxdiffusers` pipeline API (`from_pretrained`/`__call__`) is stabilizing.
- **Attention:** the live primitive is pure-MLX. The compiled Metal kernel is frozen in `research/` — it's correct but, on the GEMM-bound diffusion path, memory-not-speed (it's for KV-cached/long-context workloads).
- **First-block cache** is enabled only where it's output-safe: **on by default for Z-Image** (`cache_threshold=0.25`, near-lossless — visually identical, ~1.3×) and **excluded from FLUX**, where at 0.25 it shifts the result to a *different* (comparable-quality) image. The exact context/RoPE caches are always on for both.

## License & provenance

Apache-2.0. `mxalloy` contains no model code and no mflux lineage. `mxdiffusers/flux` was ported from / verified against [mflux](https://github.com/filipstrand/mflux) (MIT) — see `NOTICE` and `mxdiffusers/PROVENANCE.md`. `mxdiffusers/zimage` has a clean-room transformer, but currently reuses shared FLUX-derived Qwen/VAE helpers, so it is not described as wholly mflux-independent until those helpers are re-derived or split. Model weights are downloaded under their own licenses (the shipping image weights are Apache-2.0); mxalloy bundles none.
