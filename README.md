# mxalloy

**The fast, memory-lean MLX inference engine for diffusion on Apple Silicon — plus a local surface to run models on it.**

mxalloy loads, quantizes, and runs diffusion models on Apple Silicon with dramatically lower peak memory and no per-run reload. Two pieces:

- **Engine** (`pip install mxalloy`) — open, embeddable, performance-first. The thing apps and serving layers run on.
- **Surface** — a local Mac runner UI (forked from IgglePixel) that installs and runs MLX models on the engine, kept resident. A tester/on-ramp, not a consumer app.

## Why mxalloy

Apple Silicon already runs diffusion via [mflux](https://github.com/filipstrand/mflux) (open and broad, but "readability over performance" by its own README) and Draw Things (fast, but a closed app). mxalloy is the **open, embeddable, performance engine** in between:

- **Lowest peak memory.** Streaming quantized load — measured **3.89× lower peak** than mflux on FLUX.2-klein-4B (4.6 GB vs 17.9 GB), same image. 8-bit fits an 18 GB Mac where mflux can't.
- **Resident + warm.** Load once, generate many, hot-swap LoRA without reload — no MLX cold-start tax per run.
- **Quant you can trust.** Calibrated low-bit weights with published quality numbers (in progress).
- **Embeddable + stable.** Typed API, semver — built to be built on.

## Proven so far

| Config | mflux peak | mxalloy streaming peak | Reduction |
|---|---|---|---|
| 4-bit | 17.94 GB | 4.61 GB | 3.89× |
| 8-bit | 17.94 GB | 8.56 GB | 2.10× |

Full klein-4B load on an 18 GB Mac. End-to-end generation engine is in progress — see `docs/SCHEDULE.md`.

## Phase 1 Target

**FLUX.2-klein-4B** (Apache-2.0, 4-step) — ~5 GB at 4-bit, fits 18 GB. FLUX.1-schnell/dev are optional compatibility targets.

## Repository Map

- `mxalloy/quant`: quantization formats, calibration, packing, dequantization
- `mxalloy/attention`: attention execution strategies (tiled, memory-efficient)
- `mxalloy/kernels`: Metal kernel registry and launch abstractions
- `mxalloy/runtime`: device selection, resident execution, memory scheduling
- `mxalloy/models`: model adapters, starting with FLUX.2-klein
- `mxalloy/integrations`: Hugging Face / ecosystem compatibility
- `benchmarks`: repeatable performance and memory tests
- `examples`: runnable scripts
- `docs`: design brief, build plan, versioning, errors
- `experiments`: research spikes (incl. the streaming-loader benchmark)

Public API: the engine API (forming) + config dataclasses + `mxalloy.errors`. Everything else is internal and can change.

## Non-Goals (Phase 1)

- A server/daemon (serving layers exist) or a consumer app (Draw Things)
- A model-breadth race (start with klein)
- CUDA parity; training
- At-rest encryption / moderation in the local tester

## Status

Early. Streaming loader proven (3.89× lower peak than mflux). Next: promote the loader into `mxalloy`, then the resident klein generation graph, then the Mac surface. See `docs/DESIGN_BRIEF.md` and `docs/SCHEDULE.md`.
