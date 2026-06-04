# mxalloy

**The memory-lean MLX optimization layer for Apple Silicon** — fused Metal primitives, streaming quantized load, and tiled VAE that heavy models plug into.

mxalloy is the bare-metal layer beneath MLX inference: it loads, quantizes, and runs large models on Apple Silicon with dramatically lower peak memory and no per-run reload. It started as a FLUX.2 diffusion engine and is generalizing into a reusable optimization backbone — the hardest parts (a custom fused **quantized-KV attention** `mlx::core::Primitive`, **streaming quantized load**, **tiled spatial VAE**) exposed so diffusion, LLM, video-DiT, and vision-language workloads can inherit the memory wins.

Two layers:

- **Engine / infrastructure** (`pip install mxalloy`) — open, embeddable, performance-first. Custom `mlx::core::Primitive` + compiled Metal kernels (`NB_DOMAIN=mlx`, on stock MLX), the streaming loader, and tiled VAE. The thing apps, serving layers, and *other models* build on.
- **Surface** — a lean local Mac tester to dogfood the engine (refs, LoRAs, outputs, settings). A test harness, not a consumer app.

## Why mxalloy

Apple Silicon already runs diffusion via [mflux](https://github.com/filipstrand/mflux) (open + broad, but "readability over performance" by its own README) and Draw Things (fast, but a closed app). mxalloy is the **open, embeddable optimization layer** in between — and increasingly model-agnostic:

- **Lowest peak memory, proven.** Streaming quantized load — **3.89× lower peak** than mflux on FLUX.2-klein-4B (4.6 vs 17.9 GB), same image. 8-bit fits 18 GB where mflux can't.
- **Resolution decoupled from VRAM.** Tiled VAE keeps the generation peak **flat at ~14.7 GB from 1024² to 2048²**, unlocking HD + 2048² on an 18 GB Mac (≤1024² is bit-exact).
- **Spike-free quantized-KV attention.** A compiled Metal `Primitive` that inline-dequantizes int8/int4 K/V into on-chip (threadgroup) memory — the 16-bit K/V **never reach the global heap**, deleting the per-step dequant spike that standard `dequant → SDPA` pays.
- **Resident + warm.** Load once, generate many, hot-swap LoRA without reload — no MLX cold-start tax per run.
- **Embeddable on stock MLX.** Typed API, semver, a `pip install` plug-in — not a forked runtime.

## Proven so far

| Win | Result |
|---|---|
| Streaming load, 4-bit | 4.61 GB peak vs mflux 17.94 GB (**3.89×**) |
| Streaming load, 8-bit | 8.56 GB peak vs mflux 17.94 GB (**2.10×**) |
| Tiled VAE decode | gen peak **flat ~14.7 GB**, 1024² → 2048² (HD/2048² on 18 GB) |
| Fused quantized-KV attention | compiled Metal kernel, parity 5e-4 vs oracle; **flat allocation** (no dequant spike) |

Attention **speed** is a tracked work-in-progress: the v1 MMA flash kernel is correct and memory-flat but occupancy-bound (slower than MLX's tuned SDPA); the v2 speed pass is occupancy + half-MMA + `QuantizedBlockLoader`. See `docs/SCHEDULE.md`.

## What it runs / targets

- **Image suite:** FLUX.2-klein-4B (Apache-2.0, 4-step) shipping; scaling toward a **9B FLUX.2 profile** (feasibility on 18 GB). FLUX.1-schnell/dev are optional compatibility targets.
- **Infrastructure target:** the fused attention core for **KV-cached / long-context / batched** text + multimodal workloads — validated against a **Llama 3.2 3B** decode bench (`benchmarks/benchmark_kv_cache.py`).

## Repository map

- `mxalloy/loader.py`: the core model-agnostic streaming quantized loader (`load_quantized`, `QuantConfig`, `component_files`)
- `mxalloy/runtime`: device detection + memory-aware execution scheduling
- `mxalloy/attention`: fused quantized-KV attention — pure-MLX fallback (the live primitive). An experimental compiled Metal `Primitive` lives in `research/attention_kernel/` (frozen; not built by default)
- `mxalloy/kernels`: Metal kernel registry + launch abstractions
- `mxalloy/config.py`, `mxalloy/errors.py`: public config dataclasses + exception hierarchy
- `mxdiffusers/`: the diffusion framework (diffusers-style pipelines, e.g. FLUX) that runs *on* mxalloy — consumes the runtime, never the reverse
- `benchmarks`: repeatable performance + memory tests
- `research/`: frozen experiments (the compiled attention kernel)
- `surface`: lean local tester UI
- `docs`: design brief, build plan, versioning, errors
- `experiments`: research spikes (streaming loader, tiled VAE, kernel prototypes)

Public API: the engine API (forming) + config dataclasses + `mxalloy.errors`. Internals churn.

## Non-Goals (Phase 1)

- A server/daemon (serving layers exist) or a consumer app (Draw Things).
- Breadth in our *own* model adapters (start with FLUX.2) — the *primitives* are model-agnostic for others to use.
- A forked MLX; CUDA parity; training.
- At-rest encryption / moderation in the local tester.

## Status

Streaming loader, native klein-4B engine (bit-exact vs mflux), tiled VAE, and the fused quantized-KV attention `Primitive` (v1, on stock MLX) all run on 18 GB. Next: the attention speed pass + breadth (bf16/GQA/head-dims/masks), scikit-build-core packaging, the Llama-3.2-3B KV-cache proof, and 9B feasibility. See `docs/DESIGN_BRIEF.md` and `docs/SCHEDULE.md`.
