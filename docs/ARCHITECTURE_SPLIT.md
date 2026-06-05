# mxalloy Language Split Plan

This is the target architecture for mxalloy as infrastructure, not as a FLUX-only app.

## Decision

Use each language where it has durable leverage:

| Layer | Language | Owns | Does not own |
|---|---|---|---|
| Research and model bring-up | Python | model adapters, checkpoint mapping, parity tests, experiments, benchmark harnesses | resident app lifecycle, native packaging, command-buffer policy |
| Kernels and MLX primitives | Metal/C++ | custom kernels, `mlx::core::Primitive` wrappers, ABI/package boundaries, kernel parity fixtures | UI, service lifecycle, high-level model policy |
| Resident runtime | Swift | long-running engine process, memory planner, command orchestration, profiling/signposts, native app SDK/service/UI | exploratory model code, kernel math |

The north star is not to rewrite mxalloy wholesale in Swift. Keep Python as the lab bench,
keep Metal/C++ as the performance truth, and introduce Swift where process lifetime,
memory ownership, profiling, and app integration matter.

## Target Repo Shape

Current paths stay valid while the boundary hardens:

| Path | Target role |
|---|---|
| `mxdiffusers/`, `mxtts/` | Python model adapters, checkpoint mapping, and family pipelines. Internal until each family has a stable public contract. |
| `mxalloy/attention/` | Python API/oracle for attention primitives. Compiled extension is optional and detected at import time. |
| `research/attention_kernel/` | Experimental Metal/C++ attention primitive until packaged. Promote only when build, parity, and benchmark gates pass. |
| `experiments/` | Python R&D scripts. Nothing here is API. |
| `benchmarks/` | Reproducible Python benchmark entry points and published-number capture. |
| `surface/` | Temporary/local tester and dogfood UI. Keep it thin. |
| `native/` | Future Swift package/workspace for resident runtime, service, profiling, and app SDK. Not created until the Swift boundary is ready. |

## What Needs To Change

### 1. Make Python explicitly the research/model layer

- Keep model adapters in Python until their boundaries stabilize.
- Keep checkpoint conversion, safetensors mapping, parity comparisons, and quick experiments in Python.
- Move reusable benchmark utilities out of one-off `experiments/` scripts only after they have stable inputs and outputs.
- Do not add resident service concerns, long-running scheduling policy, or native app state to Python modules.

Deliverables:

- A small `mxalloy.models` adapter contract doc before adding another model family.
- Benchmark harnesses that write JSON artifacts with device, MLX version, model revision, quant mode, memory, and latency.
- Parity fixtures for every promoted model adapter.

### 2. Keep Metal/C++ focused on primitives

- Keep custom kernels as standalone primitive projects until build/ABI details are repeatable.
- Package primitives only after they have:
  - Python oracle parity tests.
  - Shape/dtype support matrix.
  - Peak-memory and latency benchmarks.
  - MLX ABI guard.
- Keep kernel APIs shape-stable and model-agnostic.

Deliverables:

- `research/attention_kernel/` build notes stay current until promotion.
- A promoted primitive moves behind `mxalloy/attention` or `mxalloy/kernels` with tests and benchmark gates.
- No kernel speed claim lands without a benchmark artifact.

### 3. Introduce Swift as the resident runtime layer

Swift should own the long-running Apple-native process, not the experimental model graph.

Initial Swift scope:

- Resident engine lifecycle: load, unload, cancel, run, report progress.
- Memory planner: model residency, scratch buffers, tiled decode budgets, KV-cache pages.
- Command orchestration: command queues, command buffers, resource labels, signposts, capture controls.
- Native integration: XPC/local service, app SDK, Keychain/config, file coordination, sandbox-aware paths.
- Profiling: `os_signpost`, Metal labels, memory-pressure logging, trace-friendly operation IDs.

Out of scope for first Swift pass:

- Rewriting FLUX model definitions.
- Replacing Python parity/benchmark workflows.
- Reimplementing MLX.
- Moving kernel math out of Metal.

Deliverables before creating `native/`:

- A Swift runtime design note with process boundary, FFI strategy, and failure modes.
- A minimal engine protocol shared by Python and Swift plans:
  - `load(model, quant, memory_mode)`
  - `unload()`
  - `generate(request)`
  - `cancel(job_id)`
  - `memory_snapshot()`
  - `set_loras(loras)`
- A decision on bridge strategy:
  - Short-term bridge: Swift host controls a Python/MLX worker process.
  - Long-term bridge: Swift runtime calls C++/MLX/Metal primitives directly where practical.

### 4. Split the current tester from the future runtime

- Keep `surface/` as a dogfood tester.
- Do not let `surface/` become the runtime architecture.
- Any resident-engine behavior proven in `surface/engine.py` should be documented before moving toward Swift.

Deliverables:

- `surface/` continues to call a thin engine manager.
- Native runtime work starts under `native/`, not by growing the FastAPI tester into a product service.

## Migration Order

1. Document the boundaries in `docs/SCHEDULE.md` and this file.
2. Finish Python benchmark/artifact discipline for current primitives and model adapters.
3. Stabilize the primitive packaging story for Metal/C++.
4. Write the Swift runtime design note.
5. Create `native/AlloyRuntime` only after the process/bridge design is reviewed.
6. Move resident lifecycle, memory planning, profiling, and native app SDK work into Swift.
7. Keep Python model adapters and experiments until a specific adapter is stable enough to expose through a model-agnostic API.

## Review Gates

Maintainer review is required before:

- Creating `native/`.
- Adding a Swift/Python or Swift/C++ bridge.
- Moving any model execution responsibility out of Python.
- Promoting a research kernel into the shipped package.
- Claiming a speed or memory win from a Swift runtime change.

## Success Criteria

This split is working when:

- Python remains fast to iterate for new models and parity.
- Metal/C++ primitives have clear build and benchmark gates.
- The resident runtime can be reasoned about as an Apple-native service with explicit memory and profiling behavior.
- The local tester uses the runtime boundary instead of becoming the runtime.
- mxalloy is easier for Mac apps and serving layers to embed without losing the Python research loop.
