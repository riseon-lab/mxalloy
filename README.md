# mxalloy

<p align="center">
  <img src="https://raw.githubusercontent.com/riseon-lab/mxalloy/main/assets/mxalloy-logo.png" alt="mxalloy logo" width="900">
</p>

**A memory-lean inference runtime for Apple Silicon, with diffusion and speech pipeline stacks built on it.**

Large models don't fail on Macs because the GPU is slow — they fail because peak memory blows past what the machine has, and unified memory makes that everyone's problem at once. mxalloy is an optimization layer on stock [MLX](https://github.com/ml-explore/mlx) that attacks peak memory directly: stream-quantized loading that never materializes the full bf16 model, adaptive memory planning that picks a plan that fits *your* machine, and resident execution so nothing reloads between runs.

One `pip` distribution provides three import packages:

| Package | What it is |
|---|---|
| `mxalloy` | The runtime: streaming quantized loader, device detection, memory-fit planner, attention primitives. Model-agnostic, stdlib-only imports, no model code. |
| `mxdiffusers` | The diffusion framework on top — a `from_pretrained(...)` → `pipe(prompt)` API in the spirit of 🤗 diffusers, organised by **architecture**, not by individual model. |
| `mxtts` | The matching speech stack. Its first pipeline adapts Miso TTS 8B via the upstream runtime while the native MLX backend is mapped. |

### Architectures

mxdiffusers targets the major modern diffusion architectures; one pipeline class covers every
checkpoint of that architecture (base models, turbo variants, finetunes). Checkpoints are
user-supplied; `MXAutoPipeline.from_pretrained(...)` routes any checkpoint to its pipeline by
reading what the checkpoint declares.

| Architecture | Pipeline | Example checkpoints | Status |
|---|---|---|---|
| SDXL | `MXSDXLPipeline` | SDXL Base, SDXL Turbo, SDXL finetunes | **Shipping** — verified vs the diffusers reference |
| FLUX.2 | `MXFluxPipeline` | FLUX.2-klein-4B | **Shipping** — verified vs the diffusers reference |
| Z-Image | `MXZimagePipeline` | Z-Image-Turbo-6B | **Shipping** — verified vs the diffusers reference |
| FLUX.1 | `MXFluxPipeline` (planned) | FLUX.1-schnell / dev / Kontext | Spec'd from the real checkpoint ([flux1/SPEC.md](mxdiffusers/flux/FLUX1_SPEC.md)) — next up |
| SD3 / SD3.5 | planned | SD3.5 Medium / Large | Blocked on the gated Stability license ([sd3/SPEC.md](mxdiffusers/sd3/SPEC.md)) |
| Qwen-Image | planned (v1.1) | Qwen-Image 20B | Needs staged execution on ≤18 GB ([qwen_image/SPEC.md](mxdiffusers/qwen_image/SPEC.md)) |

## Who this is for

- **You run diffusion/AI models locally on a Mac with 16–24 GB** and hit swap, OOMs, or per-run reload costs.
- **You build Mac-native AI tools** and want an embeddable, Apache-2.0, pure-Python layer on stock MLX — not a forked runtime or a closed app.
- **You port models to MLX** and want the loading/quantization/planning infrastructure handled so you only write the model graph and a weight-key remap.

## Measured results

18 GB M3 Pro, 4-bit, warm, FLUX.2-klein-4B (the reference is [mflux](https://github.com/filipstrand/mflux); same MLX GEMMs underneath — the difference is memory discipline). Full methodology and tables: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| | mxalloy | mflux |
|---|---|---|
| Load peak | **4.5 GB** | 17.9 GB (load-then-quantize, ~3.9× higher) |
| 512² generation (4-step) | **14.3 s / 7.5 GB** | 17.7 s / 12.5 GB |
| 1024² generation (4-step) | **44.9 s / 14.6 GB** | 54.2 s / 19.7 GB — exceeds 18 GB physical, swaps |
| Peak from 1024² → 2048² | **flat ~14.7 GB** (tiled VAE decode) | scales with pixels (~44.9 GB at 2048², OOM-class) |

- **~15–25% faster end-to-end** at both resolutions on an 18 GB machine — not from faster kernels, but from a working set that stays out of swap.
- The table's mxalloy klein rows predate the 2026-06-11 FLUX re-derivation; a same-machine, same-thermal-window A/B measured the re-derived implementation **~13% faster** (43.5 s vs 50.2 s, warm 1024² 4-step) at a **20% lower generation peak** (11.7 vs 14.65 GB) than the implementation those rows were taken with. The full benchmark refresh is a release-gate item.
- **Resolution decoupled from memory**: the pipeline's tiled VAE decode holds the generation peak flat through 2048²; at ≤1024² the single-tile path is bit-exact.
- **Z-Image-Turbo-6B**: loads in 6.2 GB (4-bit) and generates on the same 18 GB machine.
- **SDXL Base**: loads at a **2.6 GB** peak (4-bit) and generates 1024² at **3.3 s/step** with a 9.7 GB peak — vs 5.3 s/step for diffusers-on-MPS fp16 measured on the same machine; a same-latents comparison produces the same scene.
- Numbers are from `benchmarks/` scripts on the stated hardware. We do not extrapolate to hardware we haven't measured.

## Installation

```bash
pip install "mxalloy[mlx]"
```

Requires Python ≥ 3.11. The `[mlx]` extra installs MLX and is required to load and run models (Apple Silicon). A bare `pip install mxalloy` installs the mlx-free import surface — the loader/planner API types — which is importable anywhere (CI included).

## Quick start

Download a checkpoint once, then generate:

```bash
huggingface-cli download black-forest-labs/FLUX.2-klein-4B
```

```python
from mxdiffusers import MXAutoPipeline   # or MXSDXLPipeline / MXFluxPipeline / MXZimagePipeline

pipe = MXAutoPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B")  # 4-bit, resident
image = pipe("a brushed alloy sculpture, studio light", num_inference_steps=4).images[0]
image.save("out.png")
```

SDXL works the same way (30-step CFG defaults; SDXL-Turbo checkpoints want `num_inference_steps=1..4, guidance=0`):

```python
pipe = MXAutoPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
image = pipe("a corgi wearing a tiny wizard hat, oil painting", seed=42).images[0]
```

`from_pretrained` accepts a local checkpoint directory or a Hugging Face repo id resolved against your local HF cache. mxalloy is offline-first: it never downloads weights itself — if the checkpoint is missing it raises `ModelLoadError` with the exact download command.

Using the runtime directly (any MLX module, any model family):

```python
import mxalloy

files = mxalloy.component_files(model_dir, "transformer")
missing = mxalloy.load_quantized(my_mlx_module, files,
                                 remap=my_key_remap,                 # checkpoint key -> param path
                                 quant=mxalloy.QuantConfig(bits=4))
assert not missing  # full coverage check
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  your app / surface (repo-local tester UI)              │
├──────────────────────────┬──────────────────────────────┤
│  mxdiffusers             │  mxtts                       │
│  MXPipeline base +       │  MXTTSPipeline base          │
│  MXAutoPipeline router   │  miso/                       │
│  sdxl/  flux/  zimage/   │  (hybrid upstream adapter)   │
│  (model graphs, VAE      │                              │
│  tiling, schedulers,     │                              │
│  shared LoRA core,       │                              │
│  step caches)            │                              │
├──────────────────────────┴──────────────────────────────┤
│  mxalloy — the runtime                                  │
│  loader.py    streaming quantized load (the core)       │
│  runtime/     device profile + memory-fit planner       │
│  attention/   fused quantized-KV SDPA primitive         │
│  errors.py    AlloyError / ConfigurationError /         │
│               ModelLoadError                            │
├─────────────────────────────────────────────────────────┤
│  MLX (stock — not forked)                               │
└─────────────────────────────────────────────────────────┘
```

The load-bearing rule: **mxalloy never imports the model packages** — `mxdiffusers`/`mxtts` depend on mxalloy, never the reverse. This is enforced by a test (`tests/test_architecture_boundary.py`) that AST-parses every runtime module. Model-specific knowledge enters the runtime only as data (a `WorkloadSpec`, a key-remap callable), never as imports.

What each mechanism does:

- **Streaming quantized load** (`mxalloy.load_quantized`): streams safetensors one tensor at a time into an already-built resident module, quantizing eligible weights on the fly and freeing each bf16 source immediately. Peak stays near the *quantized* size instead of the full bf16 model — this is most of the 4.5 GB vs 17.9 GB gap.
- **Memory-fit planning** (`mxalloy.runtime.plan_execution`): a model family declares measured per-precision component memory and activation options; the planner deterministically picks the highest-quality precision/memory-mode plan that fits the detected working set (total memory minus OS reserve and safety margin). No-fit is reported with a reason rather than raised, so callers choose policy.
- **Tiled VAE decode** (in `mxdiffusers`): decode activations, not weights, are the >1 MP memory wall; the pipeline layer decodes in feathered tiles so peak plateaus near one 1024² decode. Single-tile (≤1024²) is bit-exact.
- **Step caching** (per model, in `mxdiffusers`): prompt-constant projections are computed once per generation (output-neutral, always on). A first-block cache (~1.3×) is on by default for Z-Image, where it is near-lossless, and deliberately excluded for FLUX, where it visibly shifts the image.
- **Quantized-KV attention** (`mxalloy.attention`): a shipped, tested primitive for KV-cached/long-context workloads. It is *not* wired into the diffusion pipelines — their 4-step txt2img path has no KV cache and attention is ~0.7% of a step; the module's own docstring says so. The memory wins above come from the loader and tiling, not attention.

## mxdiffusers: relationship to 🤗 diffusers

`mxdiffusers` mirrors the diffusers *surface* — `from_pretrained` + `pipe(prompt)` returning `.images` — because that is the API people already know. It shares no code with diffusers and has **no diffusers (or PyTorch) runtime dependency**; the denoise loops, schedulers, VAE handling, and tokenization are implemented natively on MLX, with tokenizers loaded via `transformers`.

Provenance is tracked per family and carried in [`NOTICE`](NOTICE) / [`mxdiffusers/PROVENANCE.md`](mxdiffusers/PROVENANCE.md):

- **`MXSDXLPipeline`** — SDXL Base/Turbo/finetunes (Stability AI weights, RAIL++-M). Independent MLX reimplementation derived from the Apache-2.0 diffusers/transformers references (attributed in `NOTICE`); no mflux lineage. Verified by shape coverage, text-encoder numeric parity, scheduler parity, and a same-latents image comparison against the diffusers pipeline.
- **`MXFluxPipeline`** — FLUX.2-klein-4B (Black Forest Labs, Apache-2.0 weights). Independent MLX reimplementation derived from the Apache-2.0 diffusers reference (attributed in `NOTICE`); verified by component-level numeric parity against the reference transformer and text encoder on the real weights. (Earlier versions were an mflux port — re-derived; see `PROVENANCE.md`.)
- **`MXZimagePipeline`** — Z-Image-Turbo-6B (Alibaba Tongyi, Apache-2.0 weights). Independent MLX reimplementation derived from the Apache-2.0 diffusers reference (attributed in `NOTICE`), including the shared Qwen3 text encoder and KL decoder it reuses.

We say exactly what each module's lineage is, because that is what makes the Apache-2.0 packaging trustworthy.

All shipping families support hot-swap LoRA via a shared runtime-delta core (`load_lora_weights` / `set_lora_weights` / `unload_lora_weights` — replace semantics, applied to the resident quantized model without mutating base weights). Formats: BFL/ComfyUI keys for FLUX.2, diffusers/PEFT keys for Z-Image and SDXL UNets (kohya-flattened names are a documented TODO).

## What mxalloy is not

- **Not a training framework.** Inference only.
- **Not a model zoo.** Three diffusion architectures and one speech adapter ship today, and the unit of support is the *architecture* (one pipeline class covers a family's checkpoints and finetunes); the runtime is the product, checkpoints are user-supplied.
- **Not a kernel fork.** Per-GEMM compute is stock MLX; we don't claim kernel-level speedups. The advantage is memory behaviour, and we say so.
- **Not cross-platform.** Apple Silicon is the target. The mlx-free import surface exists so libraries and CI can depend on mxalloy without mlx, not to run models elsewhere.
- **Not a GUI product.** `surface/` is a repo-local tester (model picker, LoRAs, live memory), not a shipped app.

## Repository map

- [`mxalloy/`](mxalloy/) — the runtime (public API: `load_quantized`, `QuantConfig`, `component_files`, `mxalloy.errors`, `mxalloy.runtime` planning — see [docs/VERSIONING.md](docs/VERSIONING.md))
- [`mxdiffusers/`](mxdiffusers/) — `MXPipeline` base, `MXAutoPipeline` router, shared LoRA core + `sdxl/`, `flux/`, `zimage/` families (plus `sd3/`, `qwen_image/` specs and the family's `FLUX1_SPEC.md`)
- [`mxtts/`](mxtts/) — `MXTTSPipeline` base + `miso/` (hybrid upstream adapter; native MLX backend tracked in [docs/MISO_TTS_PLAN.md](docs/MISO_TTS_PLAN.md))
- [`surface/`](surface/) — repo-local tester UI (`pip install -e ".[mlx,surface]"`)
- [`benchmarks/`](benchmarks/), [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — repeatable benchmark scripts and measured results
- [`research/`](research/) — frozen experiments (a compiled Metal attention kernel: correct, but memory-not-speed on the GEMM-bound diffusion path; not built or shipped)
- [`experiments/`](experiments/) — investigation scripts kept for reproducibility; not part of the package

## Limitations (current, honest)

- **Hardware/OS**: measured on an 18 GB M3 Pro; other configurations should scale with memory but are not yet independently verified. macOS + Apple Silicon only.
- **Batch size 1** image generation; no img2img/inpainting yet.
- **Speed ceiling is MLX's**: on GEMM-bound paths, expect memory wins (and the end-to-end speedup that comes from not swapping) — not kernel speedups.
- **FLUX guidance is inert for klein** (the checkpoint has no guidance embeddings) and Z-Image-Turbo is guidance-free; the `guidance` kwarg exists for API parity.
- **Miso TTS is a spike**: it drives the upstream PyTorch/Moshi runtime from a repo checkout (which currently pins Python 3.10 — the example bootstraps `sys.path` and cannot run against a pip-installed mxalloy in that interpreter). The native quantized MLX path is design-tracked, not shipped.
- **API stability**: the `mxalloy` core API is under the 0.x stability promise in [docs/VERSIONING.md](docs/VERSIONING.md); the `mxdiffusers` pipeline API is stabilising toward it.

## Roadmap

1. **FLUX.1 architecture** (schnell/dev/Kontext) — spec'd from the real checkpoint ([flux1/SPEC.md](mxdiffusers/flux/FLUX1_SPEC.md)). Built lineage-free: a shared T5 encoder, the verified SDXL CLIP-L, and a parameterised shared AutoencoderKL decoder.
2. **Shared T5 encoder** for FLUX.1/SD3 — joining the shared CLIP (`sdxl/clip.py`), Qwen3 (`flux/text_encoder.py`), and KL decoder (`vae_kl.py`) modules that all families now draw from.
3. **SD3/SD3.5** — once the gated Stability license is accepted ([sd3/SPEC.md](mxdiffusers/sd3/SPEC.md)); mostly MMDiT-graph work on top of the shared encoders.
4. **Qwen-Image (v1.1)** — needs staged execution (encode → free the 7B encoder → denoise the 20B transformer) to fit ≤18 GB machines; the planner's `staged` mode is the vehicle ([qwen_image/SPEC.md](mxdiffusers/qwen_image/SPEC.md)).
5. **Hybrid per-component precision and `fast` mode** in the planner — designed in [docs/EXECUTION_STRATEGY.md](docs/EXECUTION_STRATEGY.md), gated on benchmarks, not shipped until measured.
6. **Native Miso TTS backend**; **KV-cached workloads** for the quantized-KV attention primitive; **a third-party model-family guide**.

## Contributing

- `pip install -e ".[dev,mlx]"`, then `pytest` (44 tests; mlx-gated tests skip cleanly without mlx), `ruff check .`, `mypy` (strict on the core).
- CI runs four legs: lint, ubuntu-no-mlx tests, wheel-build + clean-env install smoke, macOS + mlx.
- Hard rules: `mxalloy` never imports model packages (enforced by test); new public error types ship together with the code that raises them; provenance of any ported/derived code is recorded in `PROVENANCE.md` and `NOTICE` in the same PR.
- Benchmark claims must come from `benchmarks/` scripts with hardware stated; no extrapolated numbers.

## License & provenance

Apache-2.0. The `mxalloy` runtime contains no model code and no third-party lineage. Every `mxdiffusers` model family is an independent MLX reimplementation derived from the Apache-2.0 diffusers/transformers references (attributed in [`NOTICE`](NOTICE)); see [`mxdiffusers/PROVENANCE.md`](mxdiffusers/PROVENANCE.md) for the precise lineage and verification of every module, including the FLUX.2 re-derivation history. Model weights are downloaded by you under their own licenses (both shipped image families use Apache-2.0 weights); mxalloy bundles none.
