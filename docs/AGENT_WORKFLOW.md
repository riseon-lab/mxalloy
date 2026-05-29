# Agent Workflow

Alloy is built with agent-assisted development. This doc covers how agents (and the maintainer) work together.

## The Three Hats

Collapsed from the original eight roles. A "hat" is a perspective for a focused work session — the same agent session can wear any of them.

- **Runtime Hat**: loading, MLX execution, attention scheduling, device handling, runtime memory.
- **Quant/LoRA Hat**: quantization formats, calibration, dequant math, LoRA loading, LoRA application on FP16 and INT8.
- **Surface Hat**: `enable_alloy()`, public API, integration tests, benchmarks, docs, error messages.

Switching hats is fine. Mixing them in one task usually means the task is too big — split it.

## Operating Rules

- Work from `DESIGN_BRIEF.md` and `SCHEDULE.md`. Don't add Phase 1 scope.
- New approaches start in `experiments/`. Promote into `mxalloy/` only when the module boundary is clear and there's at least a smoke test.
- Do not introduce CUDA, Triton, or NVIDIA-only assumptions into core paths.
- Public API surface (`enable_alloy`, config dataclasses) is stable in 0.1.x. Internal modules can churn freely.
- LoRA hot-swap on quantized base is the Phase 1 moat. Don't compromise it for a cleaner-looking architecture.
- Error messages are first-class. Every typed error should say: what was expected, what was received, what to do.

## Handoff Format

Each unit of work ends with:

- **Goal**: what was attempted (one sentence).
- **Files**: changed, added, deleted.
- **Commands**: how it was tested (`pytest …`, `python benchmarks/… --variant …`).
- **Results**: what works, what doesn't.
- **Risks**: known issues, fragile assumptions, things that might break under load.
- **Next**: recommended next action.

## Architecture Review Gates

Maintainer review required for changes to:

- Public API shape (`enable_alloy`, config dataclasses, `Int8QuantizedWeight`).
- Quantized weight format on disk or in memory.
- LoRA application strategy (merge vs runtime, fp16 vs quantized math, multi-LoRA composition semantics).
- Versioning policy or deprecation behavior.

## Benchmark Review Gates

Benchmark numbers must be re-captured for changes to:

- Quantization behavior.
- Attention execution.
- LoRA application.
- Memory scheduling.

Numbers go into the benchmark gallery; regressions block merge unless explicitly accepted.

## Suggested Work Tracks

For parallel sessions on the same week:

- **Track A**: MLX execution path + device handling (Runtime Hat).
- **Track B**: Quant formats, calibration, dequant math (Quant/LoRA Hat).
- **Track C**: LoRA loading, validation, runtime application (Quant/LoRA Hat).
- **Track D**: `enable_alloy` surface, error messages, public API tests (Surface Hat).
- **Track E**: Benchmark harness, reference outputs, regression tests (Surface Hat).

Each track owns a module subtree; cross-track changes go through the maintainer.

## First Concrete Assignments (Week 0)

- **Surface**: write `docs/VERSIONING.md` and `docs/ERRORS.md`. Add `diffusers` + `transformers` to `pyproject.toml`. Mark internal modules.
- **Quant/LoRA**: fix `mxalloy/quant/int8.py` grouping to per-input-channel along the input axis. Add quant grouping and dequant round-trip tests.
- **Runtime**: rename `RuntimeSchedule.estimated_peak_memory_mb` → `max_step_memory_mb` (or model liveness if cheap). Make `benchmarks/benchmark_flux.py` capture device + MLX version + a real fake-tensor timing.
