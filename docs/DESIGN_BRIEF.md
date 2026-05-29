# mxalloy Design Brief

## What mxalloy Is

mxalloy is an **open, embeddable, performance-first MLX inference engine for diffusion models on Apple Silicon** - plus a local **surface** (a lean Mac tester/runner UI) that installs and runs MLX models on top of it.

Two pieces, two kinds of moat:

1. **The engine** (`pip install mxalloy`) — the **technical moat**. Faster and dramatically leaner than the alternatives because of how it loads, quantizes, and keeps models resident. Built to be embedded by apps and serving layers.
2. **The surface** (a lean, Mac-native tester UI) - the **distribution moat**. The frictionless way to install + run MLX diffusion models on a Mac, with a curated registry of models that actually fit and work. A dogfooding/testing surface and on-ramp, not a consumer app.

Defensible position = **both together**: a competitor would need the lean engine *and* the focused local surface *and* the curated working-model registry.

## The Landscape (why this lane is open)

- **mflux** — open MLX engine, broad model zoo, library + CLI. Its own README prioritizes "readability over performance." Naive loader (measured 17.94 GB peak on klein-4B), reloads the full model per CLI run, no server. **We beat it on performance/memory; it keeps breadth.**
- **Draw Things** — closed Swift app, performance leader (own Metal FlashAttention), has an API. **We're open + embeddable; it's a closed app you can only reach via its API.**
- **Serving layers** (mlx-openai-server, MLX Studio, vMLX, …) — OpenAI-compatible API servers that run *on top of* mlx/mflux. **We're the engine they'd adopt, not a competing server.**

Our lane: **the best open, embeddable, performance engine** — what apps and those servers run on instead of mflux.

## What We've Proven

Streaming quantized load (load → quantize → free per tensor) vs the naive load-all-then-quantize mflux uses, full klein-4B on an 18 GB Mac:

| Config | mflux peak | mxalloy streaming peak | Reduction |
|---|---|---|---|
| 4-bit | 17.94 GB | 4.61 GB | 3.89× |
| 8-bit | 17.94 GB | 8.56 GB | 2.10× |

This turns "pinned at the 18 GB ceiling, thrashing" into "runs with ~13 GB to spare." 8-bit — impossible for mflux at 18 GB — is comfortable.

## Phase 1 Target

**FLUX.2-klein-4B** (Apache-2.0, 4-step): ~3.6B flow transformer + ~3.75B Qwen3 text encoder, both quantized. FLUX.1-schnell/dev are optional compatibility targets (larger LoRA ecosystem).

## The Moats, Concretely

**Engine (technical):**
1. **Performance + memory — the leader.** Streaming quantized load (proven); resident/warm execution to kill the MLX cold-start + per-run reload; Metal attention later. Fast *and* lean on every RAM tier, not just low-VRAM rescue.
2. **Calibrated quant *quality*, published.** Nobody open documents quant quality. Own "the quant you can trust" with reproducible numbers.
3. **Resident + hot-swap LoRA on a quantized model.** Load once, swap styles in ms, no reload.
4. **Embeddable, stable, typed engine API.** The thing serving layers + apps standardize on.

**Surface (distribution):**
5. **Frictionless install-and-run** for MLX models on Mac, with refs, LoRAs, outputs, local secrets, and a curated registry of models that fit Apple Silicon.

## The Local Surface

Build a small mxalloy-native local tester first. Use IgglePixel as reference material for registry shape, runner lifecycle lessons, LoRA conventions, asset handling, and `?preview` mock mode, but do not fork the full product shell for 0.1.

- **Simple frontend.** Static HTML/vanilla JS or a tiny local app shell. Black/white/solid colour system, dense controls, no glass, no heavy styling, no landing page.
- **Small FastAPI backend.** One local mxalloy engine manager; no RunPod launcher, auth gate, service worker, encryption, moderation, trainers, or CivitAI in Phase 1.
- **Actual generation workspace first.** Prompt, negative prompt, refs, model/quant/memory mode, LoRA add/remove + strength, seed, size, steps, guidance, cancel, progress, logs, memory readout, and output viewer/gallery.
- **Local secrets and paths.** Settings panel for HF token, model cache path, LoRA folder, and output folder. Store secrets backend-side only: macOS Keychain if practical, otherwise a `0600` local config file or ephemeral session mode.
- **Resident/staged modes.** Keep the hottest feasible component warm; use staged loading when unified memory is tight. The UI should expose this as a memory mode, not hide it.
- **Curated Mac registry.** Start with FLUX.2 klein 4B and 9B only; keep later model families out of the first tester until the engine path is proven.

## Phase 1 Success Criteria

**Engine:**
- Loads klein-4B with peak ≤ ~6 GB (4-bit) — beating mflux's 17.94 GB on the same machine.
- Generates a valid 1024×1024 image end-to-end (transformer + Qwen3 + VAE + flow scheduler), resident.
- Serves repeated generations warm (no per-image reload); LoRA hot-swaps without reload.
- INT8 (8-bit) path runs on 18 GB.
- Typed, documented public API; semver per `docs/VERSIONING.md`.

**Surface:**
- Local tester runs on the Mac, lists klein from a curated registry, downloads it with an HF token if needed, and generates through the mxalloy engine manager.
- UI includes refs, LoRA add/remove + strength, output viewer/gallery, logs/progress, memory status, and settings for HF token/cache/LoRA/output paths.
- `?preview` mock mode works backend-less.

## Non-Goals (Phase 1)

- A server/daemon (the serving-layer field is taken; we're the engine).
- A consumer app (that's Draw Things).
- A model-breadth race (start with klein; expand later).
- Beating Draw Things on raw Metal kernels near-term.
- At-rest encryption / moderation in the local tester.
- Training; low-bit claims without published quality numbers.

## Architectural Principles

1. **Performance and memory are the product.** Every design choice is judged on peak memory + speed first.
2. **Resident-first.** Load once, generate many, hot-swap LoRA. Cold start only on model switch.
3. **Open + embeddable.** Clean typed engine API; the thing others build on.
4. **Stable surface, fluid internals.** Public API stable per semver; internals churn.
5. **Apple-native, not Apple-only.** MLX-first, no CUDA assumptions; portable shapes at the boundary.

## Public API & Stability

See `docs/VERSIONING.md` (stable-within-minor) and `docs/ERRORS.md` (typed errors). Public surface: the engine API (forming) + config dataclasses + `mxalloy.errors`. Internals (`quant`, `attention`, `kernels`, `runtime`, `models`) churn freely.

## Definition of Success

mxalloy 0.1 succeeds if:
- It generates klein-4B on an 18 GB Mac at a peak mflux can't touch, resident and warm.
- A developer can `pip install mxalloy` and embed the engine.
- The local tester lets you (and others) install + run it with no Python wrangling.
- Published, reproducible benchmarks back the performance claim.
