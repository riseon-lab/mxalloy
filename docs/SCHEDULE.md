# mxalloy Build Plan (Phase 1 → 0.1)

Milestone-based. Supersedes the klein-only schedule: mxalloy is generalizing from a FLUX.2
engine into a reusable **Apple-Silicon optimization backbone** — streaming quantized loader,
tiled VAE, pure-MLX attention paths, and experimental Metal primitives — that heavy models
(diffusion, LLMs, video DiT, vision-language, speech) plug into. The FLUX.2 image suite is the
proving ground; reusable memory primitives are the durable, model-agnostic moat.

## Hats
- **Engine/infra**: streaming loader, fused C++/Metal primitives, quant, resident runtime, LoRA.
- **Models**: FLUX.2 image suite (klein-4B → 9B), Z-Image Turbo, Miso TTS spike; a text bench (Llama 3.2 3B) to exercise the KV-cache kernel.
- **Surface**: lean local Mac tester.
- **Bench/QA**: benchmarks, tests, published numbers.

## Language Ownership
See `docs/ARCHITECTURE_SPLIT.md` for the repo-level plan.

- **Python**: research, model adapters, checkpoint mapping, parity, experiments, benchmark harnesses.
- **Metal/C++**: kernels, MLX custom primitives, ABI/package boundaries, primitive parity fixtures.
- **Swift**: future resident runtime, memory planner, command orchestration, profiling hooks, app SDK, native service/UI.

Do not rewrite mxalloy wholesale in Swift. Introduce Swift at the runtime/process boundary
once the engine protocol and bridge strategy are reviewed.

## Non-negotiables
1. **Performance/memory lead, measured.** Every milestone keeps the win measurable. Memory wins are proven; **speed claims ship only with measured numbers** (attention speed is a tracked WIP, not yet a claim).
2. **Resident-first** — load once, run many, hot-swap LoRA.
3. **Stock MLX** — a `pip install mxalloy` plug-in on unmodified mlx, never a forked runtime.

## Done ✅
- **M0 foundations** — scaffold; per-input-channel quant + tests; errors/versioning; rename; deps.
- **M1 streaming loader** — load → quantize → free; 4/8-bit; **3.89× / 2.10×** lower peak than mflux, recorded.
- **M2 native klein-4B** — flow transformer + Qwen3 encoder + VAE + flow-match scheduler in MLX, **bit-exact vs mflux**, resident/warm. (Measured gen peak ~14.6 GB at 1024², VAE-decode-bound — *not* the old ≤6 GB target, which only described the denoise phase.)
- **Tiled VAE decode** — overlapping feathered tiles cap decode at one 1024²-equivalent tile; gen peak **flat ~14.7 GB from 1024² to 2048²**; HD + 2048² run on 18 GB; ≤1024² bit-exact.
- **Quantized-KV attention (v1)** — pure-MLX path is the shipped oracle/live primitive; the compiled Metal `Primitive` remains in `research/attention_kernel` until packaging and speed gates pass. MMA flash kernel parity exists across aligned + ragged shapes, but speed remains tracked in A.

## A — Attention core: speed + breadth
- **v2 speed**: multi-simdgroup-per-threadgroup occupancy with **shared K/V staging**, **half-input MMA** (float accumulate), `QuantizedBlockLoader`, tile tuning.
- **Breadth**: `bf16` + `fp16`; MHA + **GQA**; `head_dim` ∈ {64, 96, 128, 256}; causal masks + ragged-tail padding.
- **Done when**: parity across dtypes/layouts/masks; latency within a small factor of MLX SDPA at long context **while peak stays strictly lower** (no transient dequant).

## B — Production packaging (redistributable wheel)
- Move the extension build to **scikit-build-core**; ship one binary wheel (cibuildwheel matrix).
- Bake the MLX-ABI match into the build: nanobind pinned to MLX's (**v2.12.0** for MLX 0.31.2), `STABLE_ABI`, `NB_DOMAIN mlx` (set in `CMakeLists`, the only way the `mlx::core::array` caster is shared).
- Build-time guard: read `mlx.__version__`, select/verify the matching nanobind pin, fail loudly on mismatch.
- **Done when**: `pip install mxalloy` builds + loads the fused op against stock MLX; ABI guard covers version drift.

## C — KV-cache test bench (proof of concept): Llama 3.2 3B
- klein has no KV cache, so the kernel needs a text decode workload to prove its value.
- `benchmarks/benchmark_kv_cache.py`: 4-bit Llama 3.2 3B (~2 GB static), autoregressive decode, context **16k → 32k** (a large active KV cache).
- **Dual-path**: baseline (MLX SDPA on a dequantized cache — recurring transient spike each token) vs mxalloy fused (inline-dequant, flat allocation across context depth).
- **Done when**: published memory profiles show the baseline spike vs the flat fused path across depth, with output parity. (Needs a minimal autoregressive decode harness — scoped as part of this milestone.)

## D — FLUX.2 9B feasibility
- Stress the streaming loader + tiled VAE on a **9B FLUX.2 profile** at 4-bit on 18 GB.
- Gates: does it fit (weights + activations under the tiled-VAE ceiling)? license check (9B may be non-commercial).
- **Done when**: load + tiled decode characterized; a documented go/no-go for 18 GB.

## E — Resident engine API + LoRA hot-swap
- Stable typed engine API: load model → `generate(params)`, resident.
- LoRA on the quantized base, hot-swap without reload; multi-LoRA.
- **Done when**: load once / run many; LoRA swap < 500 ms without reload.

## F — Surface wiring
- Wire the lean tester to the **real resident engine** (currently mock generation): prompt/refs/LoRAs/dims/seed/steps, memory mode, output viewer, HF/cache/LoRA/output settings (secrets backend-side only).
- **Done when**: install + run klein from the UI via the engine manager; add a LoRA/ref; view outputs.

## F2 — Swift resident runtime plan
- Write the Swift runtime design note before creating `native/`: process boundary, engine protocol, bridge strategy, memory-planner responsibilities, profiling/signpost policy, cancellation/error model.
- Decide bridge sequence: short-term Swift host controlling a Python/MLX worker vs long-term Swift runtime calling C++/MLX/Metal primitives directly.
- Keep `surface/` as a dogfood tester; do not grow FastAPI into the production runtime.
- **Done when**: `docs/ARCHITECTURE_SPLIT.md` has an accepted design follow-up, the engine protocol is agreed, and the first `native/AlloyRuntime` scaffold has a reviewed boundary.

## G — Quant quality + memory-adaptive
- Calibrated quant (data-aware scales, mixed precision); published PSNR/LPIPS vs bf16.
- Memory-budget-adaptive config ("your RAM → fastest config that fits").

## H — Benchmarks + 0.1 release
- Reproducible suite (peak memory, latency, warm vs cold) vs mflux; API reference; tag `v0.1.0`.
- **Done when**: clean clone → install → run in < 10 min; published numbers reproduce.

## Slippage
Non-negotiables: the loader + resident generation (done) and the **fused-op memory win demonstrated end-to-end** (C). The attention **speed** pass (A v2) and **9B** (D) can slip to 0.2 if they don't pan out on 18 GB — but never ship a speed claim without measured numbers.
