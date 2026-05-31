# mxalloy Adaptive Execution Strategy

**Status: design proposal.** This document defines how mxalloy should *choose* an execution
plan for a given workload and machine, balancing memory, speed, and quality. It is grounded in
measurements on an M3 Pro / 18 GB (`experiments/precision_microbench.py`,
`experiments/perf_ceiling.py`) and is written as reusable infrastructure: FLUX.2-klein is the
proof workload, not the subject. No `mxdiffusers` work is implied here.

---

## 0. The question, and the answer

> Can mxalloy become the layer that automatically finds the *fastest viable* execution path
> for an Apple Silicon workload, balancing memory, speed, and quality?

**Yes — but the lever is not the one the framing assumes.** The measurements show that on
Apple Silicon, at diffusion sequence lengths, *precision does not buy speed*. Compute is the
wall, and every precision sits against it. Therefore:

- **Precision is a memory knob, chosen by the memory budget.** Within a budget you pick the
  *fastest precision that fits* — which is the **highest** precision that fits, not the lowest.
- **Speed comes from doing less work** (steps, blocks, tokens), not from cheaper arithmetic.
- The adaptive layer's job is a **constraint solve**: fit the model in the working set at the
  best quality, then spend any leftover memory on *speed* by raising precision, and spend any
  explicit "fast" budget on *work reduction*.

That reframing is the whole strategy. The rest of this document makes it concrete and testable.

---

## 1. What the hardware actually does (measured)

`experiments/precision_microbench.py`, M3 Pro, the four GEMM shapes that dominate a klein
denoise step, at M = 4096 image tokens (1024²):

| GEMM (in→out)         | bf16 | fp16 | int8 | int4 | int8 vs bf16 | int4 vs bf16 |
|-----------------------|-----:|-----:|-----:|-----:|-------------:|-------------:|
| attn_qkv 3072→9216    | 3.78 | 4.08 | 3.51 | 3.62 | **1.08×**    | 1.05×        |
| attn_out 3072→3072    | 4.08 | 4.07 | 3.37 | 3.56 | **1.21×**    | 1.15×        |
| mlp_up 3072→12288     | 4.35 | 4.25 | 3.53 | 3.59 | **1.23×**    | 1.21×        |
| mlp_down 12288→3072   | 3.82 | 3.76 | 3.73 | 3.81 | 1.03×        | 1.00×        |

(TFLOP/s; "×" is latency relative to bf16, lower is faster.)

**Readings:**

1. **Compute-bound regime.** Every precision lands at ~3.5–4.4 TFLOP/s — the M3 Pro's bf16
   matmul ceiling (~4 TFLOP/s). Nothing is memory-starved at these shapes/sizes.
2. **bf16 ≈ fp16.** Within ±2–7%, sometimes fp16 marginally ahead. fp16 is *not* a speed win;
   bf16 is the safer default (wider dynamic range, same speed).
3. **Quantized matmul is slower, never faster** — 0–23% slower than bf16. The inline dequant is
   added compute in a regime that has no spare compute. **There is no path to making int4/int8
   execution beat bf16 on this hardware for this workload.** The ceiling *is* the bf16 GEMM.
4. **int4 ≈ int8 on speed.** int4 is occasionally a hair faster than int8 (less weight traffic).
   So int8 costs 2× int4's memory for the *same* speed — its only advantage is quality. int8 is
   a narrow niche, not a tier you reach for by default.
5. **The slowdown is shape-dependent.** Compute-bound shapes (large output dim: `mlp_up`,
   `attn_out`) pay 1.15–1.23×; the memory-bound `mlp_down` pays ~nothing. **This is exactly the
   seam hybrid precision exploits**: leaving the compute-bound projections in bf16 (when memory
   allows) recovers most of the loss; quantizing the memory-bound ones is nearly free.

**Cross-check with `perf_ceiling.py`:** the full klein step is GEMM-bound, attention is ~0.7%,
and we already sit at ~90% of the achievable step time at 1024²/4-step. Consistent: micro-opts
on execution are spent; the remaining speed is in *work*, *resolution*, and *precision-if-it-fits*.

### Bytes per parameter (the memory side of the trade)

| precision | bytes/param | relative |
|-----------|------------:|---------:|
| bf16 / fp16 | 2.00 | 1.00× |
| int8 (g64 + scales) | ~1.06 | 0.53× |
| int4 (g64 + scales) | ~0.56 | 0.28× |

So the trade is stark: int4 is ~10–15% slower than bf16 on the hot GEMMs but uses ~3.6× less
weight memory. **On a memory-starved machine you take that trade because bf16 simply will not
fit; on a roomy machine you decline it because bf16 is both faster and higher quality.**

### Why FP8 is out of scope (decision, with reason)

MLX 0.31 has no fp8 matmul. Even if it did: fp8 is compute-bound too, so it would *at best* match
bf16 speed (point 3), while its memory (1 byte/param) sits between int8 and bf16 — strictly
dominated by int4 on memory and by bf16 on speed/quality. There is no quadrant where fp8 wins
this workload. **Do not pursue fp8.** Revisit only if a future MLX ships hardware-accelerated
fp8 *and* a memory-bound workload appears (e.g. LLM decode), not for diffusion.

---

## 2. Deliverable 1 — Hardware-aware execution strategy

The strategy is a **memory-first constraint solve** with two budgets:

```
working_set = total_unified_memory − os_reserve − safety_margin
```

The planner picks the **highest-quality precision whose estimated peak ≤ working_set**, then
spends any explicit `fast` budget on work reduction. Defaults are deterministic; fast modes are
opt-in.

### Strategy table (defaults per machine class, klein-class image workload)

| Class            | `quality`            | `balanced` (default)      | `fast` (opt-in)                              |
|------------------|----------------------|---------------------------|----------------------------------------------|
| **18 GB** (M-Pro)| int4, full steps     | **int4, full steps**      | int4 + step/block/token reduction, ↓res      |
| **36–48 GB**     | int8 (or hybrid)     | **hybrid: bf16 hot + int4**| hybrid + work reduction                     |
| **64 GB** (Max)  | bf16, full steps     | **bf16, full steps**      | bf16 + work reduction                        |
| **128 GB+** (Ultra)| bf16, full steps   | **bf16, full steps**      | bf16 + work reduction (or bigger model/batch)|

Notes:
- 18 GB has **one viable precision (int4)** for klein — the table's three modes differ only in
  *work*, not precision. This is the honest shape of the constraint, and it's what ships today.
- The 36–48 GB row is where **hybrid** earns its place: not enough room for full bf16, enough to
  un-quantize the compute-bound projections (Section 4) and recover most of the 10–15%.
- 64 GB+ runs full bf16: fastest *and* highest quality. Quantization there is pointless for klein
  (it would only slow things down). It re-enters only for a model too big to fit in bf16.
- The table is **per-workload**: a 9B model shifts every row up (9B bf16 needs ~34 GB weights, so
  64 GB becomes the "hybrid/int8" row and 18 GB stays int4-only). The planner computes this from
  param counts; the table is just the planner's output for the klein case.

### The model is reusable, not FLUX-shaped

The planner consumes an abstract `WorkloadSpec` (param counts per component, sequence length,
step count, an activation-memory model) and a `DeviceProfile`. A model adapter *supplies* those;
the core decides. Nothing in the planner mentions FLUX. An LLM adapter would hand it a different
`WorkloadSpec` (KV-cache-dominated, memory-bound) and get a different — correct — answer.

---

## 3. Deliverable 5 — Proposed config structure

Extends what already exists (`mxalloy/config.py`, `mxalloy/runtime/device.py`,
`mxalloy/runtime/scheduler.py`) rather than greenfielding. All in the **core**, model-agnostic.

```python
# mxalloy/runtime/device.py  — extend the existing AppleSiliconDevice
@dataclass(frozen=True, slots=True)
class DeviceProfile:
    machine: str                      # "arm64"
    is_apple_silicon: bool
    chip: str | None                  # "M3 Pro" if detectable, else None
    total_memory_gb: float            # mx.metal.device_info() / sysctl hw.memsize
    working_set_gb: float             # total − os_reserve − safety_margin (the budget)

Mode = Literal["quality", "balanced", "fast"]
Precision = Literal["bf16", "int8", "int4"]   # fp16 omitted: no speed win vs bf16 (§1)

# mxalloy/config.py  — the workload description a model adapter supplies
@dataclass(slots=True)
class ComponentSpec:
    name: str                         # "transformer", "text_encoder", "vae"
    params: int                       # parameter count
    quantizable: bool = True          # VAE → False (decode is the activation peak, not weights)

@dataclass(slots=True)
class WorkloadSpec:
    components: list[ComponentSpec]
    seq_len: int                      # tokens at target resolution
    steps: int
    activation_peak_gb: float         # measured/estimated transient peak (e.g. VAE decode)

# the planner OUTPUT — what the engine executes
@dataclass(slots=True)
class ExecutionStrategy:
    precision: dict[str, Precision]   # per component, e.g. {"transformer": "int4", ...}
    hybrid_bf16_layers: tuple[str, ...] = ()   # layer-name globs kept in bf16 (§4)
    vae_tile_latent: int | None = 128
    steps: int = 4
    skip_blocks: tuple[int, ...] = () # single-stream blocks to skip at late timesteps (§4)
    token_merge: float = 0.0          # fraction merged in fast mode (§4)
    estimated_peak_gb: float = 0.0    # the fit check that produced this plan
```

The existing `RuntimeSchedule` / `ExecutionStep.estimated_memory_mb` become the peak estimator
that feeds `estimated_peak_gb`. `QuantizationConfig` is subsumed by `precision` per component.

---

## 4. Deliverable 6 — Automatic decision rules

Deterministic, with an explicit fit check and escalation ladder. Pseudocode:

```
plan(device, workload, mode) -> ExecutionStrategy:
    budget = device.working_set_gb

    # 1. Precision ladder, highest quality first; take the first that fits.
    for precision in [bf16, int8, int4]:            # quality-descending
        est = estimate_peak(workload, precision)     # Σ params·bytes/param + activation_peak
        if est <= budget:
            chosen = precision; break
    else:
        chosen = int4                                # smallest; if it still doesn't fit → see ladder

    # 2. Spend leftover memory on SPEED (only meaningful between int4 and bf16).
    if chosen == int4 and budget − est > headroom_for_hybrid:
        hybrid = pick_compute_bound_layers(workload, budget − est)   # un-quantize hottest GEMMs
    else:
        hybrid = ()

    # 3. Mode shapes WORK, not precision.
    if mode == "fast":
        steps      = max(3, workload.steps − 1)
        skip       = late_timestep_single_blocks()   # opt-in, experimental
        token_merge= 0.3                              # opt-in, experimental
    elif mode == "quality":
        steps = workload.steps                        # no work reduction, highest precision that fits
    else:  # balanced (default)
        steps = workload.steps; skip = (); token_merge = 0.0

    # 4. If even int4 + tiled VAE exceeds budget: escalate compression, then degrade gracefully.
    #    ladder: shrink vae_tile_latent → lower resolution → (report) cannot fit.
    return assemble(chosen, hybrid, steps, skip, token_merge, fit_checked=True)
```

**Key rules, justified by §1:**
- *Highest precision that fits*, not lowest — bf16 is fastest and best, so only drop precision
  when forced by the budget.
- *int8 is never chosen for speed* — same speed as int4, double the memory. It appears only as a
  quality step on machines where int4 quality is unacceptable *and* bf16 doesn't fit (the thin
  36–48 GB band), and even there hybrid-bf16 is usually the better spend.
- *Determinism*: `quality` and `balanced` are bit-reproducible (fixed seed → fixed image). Only
  `fast` introduces approximation, and only when explicitly requested.

---

## 5. Deliverables 2–4 — Benchmark plans

Each plan states the **question**, the **method**, and the **decision** the result drives.

### 5.1 Precision benchmark (Deliverable 2)

- **Question:** does the GEMM microbench predict the *full-step* and *end-to-end* cost, and what
  is the real peak at each precision?
- **Method:** klein 1024²/4-step, measure step time + `mx.get_peak_memory()` + load time + a
  quality metric (PSNR/SSIM vs a bf16 reference on a fixed seed) for {int4, int8, hybrid, bf16}.
  bf16 will OOM on 18 GB — run the bf16/int8 points on a roomier machine or on the 9B-as-stress
  proxy; record the OOM as data.
- **Decision:** confirms (or corrects) the per-machine rows in §2 and calibrates `estimate_peak`.
- **Status:** the GEMM layer is **done** (§1). The full-step/quality layer is next-to-run.

### 5.2 Hybrid precision benchmark (Deliverable 3)

- **Question:** how much of the 10–15% does un-quantizing the compute-bound layers recover, and
  what does each cost in memory? Which layers are worth it?
- **Method:** start int4-everywhere; promote to bf16 in this order (most compute-bound first, per
  §1): single-stream `qkv_mlp` proj → single-stream `to_out` → double-stream MLP → attn out
  projections. After each promotion measure Δ step-time and Δ peak. Also test the inverse for
  encoders/VAE (already bf16) — confirm they shouldn't be quantized.
- **Decision:** produces `pick_compute_bound_layers()` — the ranked promotion list and its
  memory cost curve, i.e. the 36–48 GB "hybrid" row.
- **Risk to watch:** mixing precisions adds dequant/cast boundaries; verify the recovered compute
  time isn't eaten by conversion overhead.

### 5.3 Work-reduction benchmark (Deliverable 4) — *the real speed lever*

All **experimental, opt-in**. Question for each: speed gain vs quality cost (PSNR/SSIM + eyeball)
vs a 4-step int4 reference.

- **Step count:** 4 → 3 → 2 steps. (Biggest, simplest lever; klein is already a few-step model.)
- **Single-stream block skipping at late timesteps:** the 20 single blocks dominate; skip a
  subset on the last 1–2 steps where the latent is nearly settled (TeaCache/▵-style intuition).
- **Token merging / reduction:** merge similar image tokens before the GEMMs, unmerge after
  (ToMe-style). Directly cuts M in the §1 GEMMs — the only thing that moves a compute-bound wall.
- **Block-output reuse (TeaCache-style):** cache and reuse block outputs across adjacent steps
  when the timestep delta is small.
- **Decision:** defines the contents of `fast` mode and the quality floor at which each is enabled.

---

## 6. Deliverable 7 — Risks and likely quality tradeoffs

| Lever | Quality impact | Risk | Mitigation |
|-------|----------------|------|------------|
| int4 weights | ~moderate (the OOM-only lever; visible vs bf16) | the floor of 18 GB quality | it's forced on 18 GB; offer int8/hybrid where memory allows |
| int8 weights | near-lossless (<1%, >40 dB) | none on quality; 2× int4 memory for no speed | only as a quality step, never for speed |
| bf16 | reference quality | memory | only when it fits |
| hybrid precision | between int4 and bf16 | cast/dequant boundary overhead can eat the gain | benchmark each promotion (§5.2); keep boundaries coarse (whole layers) |
| fewer steps (4→3→2) | degrades fast below model's design point | structure/detail loss | opt-in; per-model floor; default stays 4 |
| block skipping | subtle texture/detail loss | wrong blocks → artifacts | only late timesteps; opt-in; verified per schedule |
| token merging | softening, lost fine detail | merge metric sensitivity | conservative fraction; opt-in |
| block reuse | ghosting if delta too large | threshold tuning | conservative threshold; opt-in |

**Determinism guarantee:** `quality`/`balanced` are deterministic (fixed seed → fixed bytes).
Tiled VAE is already bit-exact ≤1024² and feathered above (per-tile GroupNorm drift handled). All
quality-degrading levers live behind `fast` and are individually toggleable.

**Strategy-selection risk:** a wrong memory estimate → OOM (too optimistic) or needless quality
loss (too pessimistic). Mitigation: conservative `safety_margin`, the escalation ladder (§4 step
4) degrades gracefully rather than crashing, and a CI invariant test asserts the estimate bounds
the real peak on reference workloads.

---

## 7. Deliverable 8 — What to test first (prioritized by leverage)

1. **✅ Precision GEMM microbench** — *done* (§1). Settled the central question: quant = memory,
   not speed; bf16 is the ceiling; fp8 is out.
2. **Work-reduction: 4→3→2 steps on klein** (highest speed leverage, simplest, no memory cost).
   This is where the real wins are and it's runnable on 18 GB today. Do this next.
3. **Full-step precision + peak + quality** (§5.1) to calibrate `estimate_peak` and confirm the
   strategy table. Cheap, anchors the planner.
4. **Hybrid promotion curve** (§5.2) — only matters once a 36–48 GB machine exists, but the
   *method* can be validated on 18 GB by measuring step-time deltas of promoting one layer (even
   if peak then exceeds budget, the timing delta is the data).
5. **Block skipping / token merge** (§5.3) — higher effort, defer until step-count results show
   how much headroom `fast` mode needs.

**Build order for the layer itself:** `DeviceProfile` (real memory detection) → `estimate_peak`
(calibrated by test 3) → `plan()` with the precision ladder (deterministic, ships as `balanced`)
→ `fast` mode work-reduction (opt-in) → hybrid (when hardware justifies it).

---

## 8. Positioning (constraints honored)

- **Reusable infrastructure.** The planner, `DeviceProfile`, `WorkloadSpec`, and
  `ExecutionStrategy` live in `mxalloy` core and mention no model. FLUX.2-klein supplies a
  `WorkloadSpec` and consumes a strategy — exactly as a future LLM or video adapter would.
- **FLUX is the proof, not the identity.** Every measurement here is a klein number, but every
  *rule* is expressed over abstract params/seq/steps/budget.
- **Safe, deterministic defaults; experimental fast modes opt-in.** `balanced` = today's
  behaviour (int4, 4-step, deterministic). Nothing approximate runs unless asked.
- **No `mxdiffusers` yet, no repo-wide overfit.** This is a core-layer design; the FLUX engine is
  the first consumer, not the template.

**One-line summary:** mxalloy's adaptive layer is a *memory-budget constraint solver* — fit the
model at the best precision the machine allows, then buy speed with work reduction, not with
cheaper arithmetic. The hardware says so.
