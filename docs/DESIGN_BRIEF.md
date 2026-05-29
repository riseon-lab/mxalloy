# mxalloy Design Brief

## What mxalloy Is

mxalloy is the **memory-lean MLX optimization layer for Apple Silicon** — the bare-metal
infrastructure beneath inference: custom fused Metal primitives, a streaming quantized
loader, and tiled spatial VAE that load, quantize, and run large models with dramatically
lower peak memory and no per-run reload. It began as a FLUX.2 diffusion engine and is
generalizing into a reusable, **model-agnostic** backbone — diffusion, LLM, video-DiT, and
vision-language workloads inherit the memory wins by plugging into the same primitives.

Two layers:

1. **Engine / infrastructure** (`pip install mxalloy`) — the **technical moat**. The hardest
   bare-metal optimizations as reusable pieces: a custom fused **quantized-KV attention**
   `mlx::core::Primitive` + Metal kernels (`NB_DOMAIN=mlx`, on stock MLX), the streaming
   loader, and tiled VAE. Embeddable by apps, serving layers, and other models.
2. **Surface** — a lean local Mac **tester** that runs models on the engine (refs, LoRAs,
   outputs, local settings). A dogfooding harness and on-ramp, not a consumer product.

Defensible position: the **optimization layer itself**. A competitor would have to
re-implement the hardest Metal/quant work — streaming quantized load, tiled VAE, and a
spike-free fused attention primitive — and keep it embeddable on unmodified MLX.

## The Landscape (why this lane is open)

- **mflux** — open MLX engine, broad model zoo. Its own README prioritizes "readability over
  performance." Naive loader (17.94 GB peak on klein-4B), reloads per run, no server.
  **We beat it on performance/memory; it keeps breadth.**
- **Draw Things** — closed Swift app, performance leader (own Metal FlashAttention), has an
  API. **We're open + embeddable; it's a closed app reachable only via its API.**
- **Serving layers** (mlx-openai-server, MLX Studio, …) — API servers that run *on top of*
  mlx/mflux. **We're the engine they'd adopt, not a competing server.**
- **Heavy-model authors** (video DiT like Wan, high-context LLMs/VLMs) — fighting unified-memory
  ceilings on Apple Silicon. **We're the optimization layer they swap allocation-heavy
  attention/load paths for.**

Our lane: **the best open, embeddable, performance optimization layer** — what apps, servers,
and other models build on instead of hand-rolling Metal/quant.

## What We've Proven

On an 18 GB Mac, on **stock MLX**:

| Win | Result |
|---|---|
| Streaming quantized load, 4-bit | 4.61 GB peak vs mflux 17.94 GB (**3.89×**) |
| Streaming quantized load, 8-bit | 8.56 GB peak vs mflux 17.94 GB (**2.10×**) |
| Tiled VAE decode | generation peak **flat ~14.7 GB, 1024² → 2048²** (HD + 2048² fit; ≤1024² bit-exact) |
| Fused quantized-KV attention | compiled Metal `Primitive`, parity **5e-4** vs oracle; **flat allocation** (no global dequant spike) |

Streaming load alone turns "pinned at the 18 GB ceiling, thrashing" into "runs with ~13 GB to
spare." The fused attention kernel's **speed** is a tracked WIP (v1 is correct + memory-flat
but occupancy-bound vs MLX's tuned SDPA); the **memory** win is the proven claim.

## Targets

- **Image suite:** FLUX.2-klein-4B (Apache-2.0, 4-step) shipping, bit-exact vs mflux; scaling
  toward a **9B FLUX.2 profile** (feasibility on 18 GB; license gate). FLUX.1-schnell/dev are
  optional compatibility targets.
- **Infrastructure:** the fused attention core for **KV-cached / long-context / batched** text +
  multimodal workloads, validated against a **Llama 3.2 3B** decode bench.

## The Moats, Concretely

**Engine (technical):**
1. **Performance + memory — the leader.** Streaming quantized load, tiled VAE, resident/warm
   execution. Proven memory wins; fused-attention speed in progress.
2. **Model-agnostic fused primitives.** The attention `Primitive`, loader, and VAE are reusable
   across model families — the durable, hard-to-copy moat.
3. **Calibrated quant *quality*, published.** Nobody open documents quant quality. Own "the quant
   you can trust" with reproducible PSNR/LPIPS.
4. **Resident + hot-swap LoRA on a quantized base.** Load once, swap styles in ms, no reload.
5. **Embeddable, stable, typed API on stock MLX.** A `pip install` plug-in, not a forked runtime.

**Surface (tester / on-ramp):**
6. **Frictionless install-and-run** for MLX models on Mac — refs, LoRAs, outputs, local secrets,
   curated registry. A dogfood harness, not a product moat.

## The Local Surface

A small mxalloy-native local tester (see `docs/UI_PLAN.md`): one generation workspace, static
HTML/vanilla JS + a small FastAPI backend over one resident engine manager. Black/white/dense,
no landing page. Local-only, single-user. IgglePixel is reference material, not a fork. Secrets
stay backend-side (Keychain or `0600` config); `?preview` mock mode works backend-less.

## Phase 1 Success Criteria

**Engine:**
- Loads klein-4B at a peak mflux can't touch (4.61 GB at 4-bit vs 17.94 GB), resident/warm. ✅
- Generates a valid image end-to-end (transformer + Qwen3 + VAE + flow scheduler), bit-exact. ✅
- Tiled VAE keeps generation feasible to 2048² on 18 GB (peak flat ~14.7 GB). ✅
- The fused quantized-KV `Primitive` builds + runs on stock MLX, with a demonstrated **flat
  allocation vs the dequant-spike baseline** (KV-cache bench).
- 8-bit path runs on 18 GB; typed, documented public API; semver per `docs/VERSIONING.md`.

**Surface:**
- Local tester lists klein from a curated registry, downloads with an HF token, and generates
  through the resident engine manager; refs, LoRA add/remove + strength, outputs, settings.

## Non-Goals (Phase 1)

- A server/daemon (the serving-layer field is taken; we're the engine).
- A consumer app (that's Draw Things).
- Breadth in our *own* model adapters (start with FLUX.2) — the *primitives* stay model-agnostic.
- A forked MLX runtime; CUDA parity; training.
- At-rest encryption / moderation in the local tester.
- Low-bit or speed claims without published numbers.

## Architectural Principles

1. **Performance and memory are the product.** Every choice judged on peak memory + speed first;
   speed claimed only when measured.
2. **Resident-first.** Load once, run many, hot-swap LoRA. Cold start only on model switch.
3. **Model-agnostic primitives.** The hard optimizations are reusable across model families.
4. **Open + embeddable on stock MLX.** Clean typed API; a plug-in, never a forked runtime.
5. **Stable surface, fluid internals.** Public API stable per semver; internals churn.
6. **Apple-native, not Apple-only.** MLX-first, no CUDA assumptions; portable shapes at the boundary.

## Public API & Stability

See `docs/VERSIONING.md` (stable-within-minor) and `docs/ERRORS.md` (typed errors). Public
surface: the engine API (forming) + config dataclasses + `mxalloy.errors`. Internals
(`quant`, `attention`, `kernels`, `runtime`, `models`) churn freely.

## Definition of Success

mxalloy 0.1 succeeds if:
- It runs FLUX.2-klein on an 18 GB Mac at a peak mflux can't touch, resident and warm. ✅
- A developer can `pip install mxalloy` and embed the engine — and the fused primitives — on stock MLX.
- The fused quantized-KV attention path demonstrates a flat allocation profile a dequant-then-SDPA
  baseline can't match (the infrastructure proof point).
- The local tester lets you (and others) install + run models with no Python wrangling.
- Published, reproducible benchmarks back every performance claim.
