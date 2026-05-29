# mxalloy Build Plan (Phase 1 → 0.1)

Milestone-based. The old week-by-week FP16/FLUX.1 schedule is superseded by the klein pivot and the streaming-loader result. Order matches the agreed sequence: revise → promote → build.

## Hats
- **Engine**: loader, MLX execution, quant, attention, resident runtime, LoRA.
- **Surface**: the Mac runner UI (forked IgglePixel), registry, workspaces.
- **Bench/QA**: benchmarks, tests, published numbers.

## Non-negotiables
1. **Performance/memory lead** — every milestone keeps the peak-memory + speed win measurable.
2. **Resident-first engine** — load once, generate many, hot-swap LoRA.

## M0 — Foundations ✅ (done)
Scaffold; per-input-channel INT8 quant + dequant + tests; exception hierarchy; versioning/error docs; benchmark harness; `mxalloy` rename; klein committed as target; deps. Streaming loader proven in `experiments/` (3.89× / 2.1× lower peak than mflux).

## M1 — Streaming quantized loader (promote the proven moat)
- Promote `experiments/streaming_quant_load.py` into `mxalloy` as a real module (mlx-gated).
- Streaming load → quantize → free for safetensors; per-input-channel; 4/8-bit.
- mlx-gated test (skips without mlx) + a benchmark recording peak vs naive.
- **Done when**: `mxalloy` exposes a streaming quantized loader with the ~3.9× peak win as a recorded benchmark.

## M2 — klein graph, resident-first (engine can generate)
- Implement klein-4B in MLX: flow-transformer forward, Qwen3 text encoder, VAE, flow-match scheduler.
- Load weights via the streaming loader; keep the model **resident** (load once, generate many).
- **Done when**: `mxalloy` generates a valid 1024×1024 klein image end-to-end at peak ≤ ~6 GB (4-bit), with repeated generations warm.

## M3 — Resident engine API + LoRA hot-swap
- Stable, typed engine API: load model → `generate(params)` → image, resident.
- LoRA on the quantized base, applied at runtime; hot-swap without reload; multi-LoRA.
- **Done when**: load once, generate many; swap LoRAs in < 500 ms without reload; INT8 path runs on 18 GB.

## M4 — Mac surface (fork IgglePixel)
- Fork the IgglePixel shell + registry; runner = **resident mxalloy**; one image workspace.
- Curate registry to Mac-fitting MLX models; unified-RAM tiers; drop encryption/moderation; keep `?preview`.
- **Done when**: locally install + run klein from the UI via the resident engine; dogfooding happens here.

## M5 — Quant quality + memory-adaptive
- Calibrated quant (data-aware scales, mixed precision); published PSNR/LPIPS vs bf16.
- Memory-budget-adaptive config ("your RAM → fastest config that fits").
- **Done when**: published quant-quality numbers; auto-config picks a fitting, fast setup.

## M6 — Benchmarks + 0.1 release
- Reproducible benchmark suite (peak memory, latency, warm vs cold) vs mflux on reference hardware.
- API reference, README, known limitations; tag `v0.1.0`.
- **Done when**: clean clone → install → generate in < 10 min; published numbers reproduce.

## Slippage
Non-negotiables are **M1** (loader) and **M2** (resident generation). M5 (quant quality) and parts of M4 can move to 0.2. Do not ship without the performance/memory win demonstrated end-to-end.
