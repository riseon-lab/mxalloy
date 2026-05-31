# mxalloy Performance Investigation - Phase 2

Measurement date: 2026-05-30
Device reported by MLX: Apple M3 Pro, `applegpu_g15s`, 19.33 GB memory, 14.30 GB recommended working set.
Workload: FLUX.2-klein, 1024x1024, 4096 image tokens + 512 text tokens, 4 denoise steps; measurements below are for one denoise step.

## Methodology

Harness: `experiments/perf_phase2.py`.

- `e2e`: times lazy transformer graph construction, scheduler graph construction, and `mx.eval` synchronization separately.
- `timeline`: replays the transformer with `mx.eval` boundaries at named model components so rows sum to the measured instrumented wall time.
- `linears`: benchmarks representative largest linear layers and compares quantized matmul with full-weight `mx.dequantize` and dense bf16 matmul baselines.
- Metal: `xcrun xctrace` attached after model load/warmup using `Metal Application` and `GPU` instruments.

Raw result artifacts:

- `/private/tmp/mxalloy_phase2_int4_e2e.json`
- `/private/tmp/mxalloy_phase2_int4_timeline.json`
- `/private/tmp/mxalloy_phase2_int4_linears.json`
- `/private/tmp/mxalloy_phase2_int8_e2e.json`
- `/private/tmp/mxalloy_phase2_int8_linears.json`
- `/private/tmp/mxalloy_phase2_bf16_e2e.json`
- `/private/tmp/mxalloy_phase2_bf16_linears.json`
- `/private/tmp/mxalloy_phase2_int4_metalapp.trace`
- `/private/tmp/mxalloy_phase2_int4_gpu.trace`

## Missing-Time Accounting

Uninstrumented int4 median over 3 warm iterations:

| Component | Time |
|---|---:|
| Transformer graph construction | 10.497 ms |
| Scheduler graph construction | 0.045 ms |
| `mx.eval` sync/device wait | 10,765.482 ms |
| Total | 10,803.348 ms |

Instrumented transformer timeline:

| Component | Time |
|---|---:|
| Single-stream `parallel_attn_mlp` blocks | 8,437.480 ms |
| Double-stream MLP | 1,132.496 ms |
| Double-stream attention module | 1,008.271 ms |
| Residual/gate elementwise work | 99.773 ms |
| Single-stream norm/mod | 52.562 ms |
| Double-stream norm/mod | 26.133 ms |
| Conditioning/modulation | 14.345 ms |
| Input projections | 12.668 ms |
| Output projection/norm | 5.444 ms |
| RoPE | 3.160 ms |
| Layout conversions | 0.906 ms |
| Host/timestep orchestration | 0.272 ms |
| Timer/loop residual | 2.562 ms |
| Transformer total | 10,796.072 ms |
| Scheduler tail | 0.805 ms |

This accounts for 10,796.877 ms of measured timeline wall-clock. The earlier ~6.5 s "block execution" estimate undercounted block work; the missing time is not external overhead.

Metal Application trace for a warmed int4 step:

- Helper-measured step: 10,970.409 ms.
- Command-buffer submission span: 10,757.624 ms.
- Command-buffer submissions: 710.
- Command buffers with encoders: 373.
- Total app-side encoder time: 27.705 ms.
- Completion-handler time: 24.147 ms.

Conclusion: command submission/encoding and completion handlers are measured in tens of milliseconds, not seconds.

## Quantization Tradeoff

All modes used the same 1024x1024 workload and one warmup.

| Mode | Median step | Peak GB | Active GB | Logical linear throughput |
|---|---:|---:|---:|---:|
| bf16 | 9,424.160 ms | 16.213 | 14.841 | 3.003 TFLOP/s |
| int4 | 10,803.348 ms | 5.641 | 4.322 | 2.619 TFLOP/s |
| int8 | 10,980.948 ms | 9.319 | 7.999 | 2.577 TFLOP/s |

Logical linear work from model shapes is 28.297 TFLOPs per denoise step. Int4 and int8 reduce memory, but they do not improve latency on this workload. Bf16 is ~12.8% faster than int4 while using ~10.6 GB more peak memory.

Observed load/prep times from benchmark stdout:

- int4: 5.72 s, 5.31 GB peak after load.
- int8: 6.12 s, 8.79 GB peak after load.
- bf16: 8.61 s, 15.45 GB peak after load.

## Largest Linear Layers

Representative first-block measurements:

| Layer | Mode | Time | TFLOPs | TFLOP/s | Dense bf16 baseline | Weight dequant |
|---|---:|---:|---:|---:|---:|---:|
| `single0.to_qkv_mlp_proj` | int4 | 207.829 ms | 0.783 | 3.766 | 172.823 ms | 3.224 ms |
| `single0.to_qkv_mlp_proj` | int8 | 210.496 ms | 0.783 | 3.719 | 185.230 ms | 4.147 ms |
| `single0.to_qkv_mlp_proj` | bf16 | 173.169 ms | 0.783 | 4.520 | n/a | n/a |
| `double0.ff.linear_in` | int4 | 126.169 ms | 0.464 | 3.676 | 102.030 ms | 2.248 ms |
| `double0.ff.linear_in` | bf16 | 108.483 ms | 0.464 | 4.276 | n/a | n/a |
| `single0.to_out` | int4 | 95.805 ms | 0.348 | 3.631 | 86.822 ms | 2.405 ms |
| `single0.to_out` | bf16 | 78.118 ms | 0.348 | 4.453 | n/a | n/a |

SwiGLU activation for `double0.ff.linear_in`: 5.39 ms int4, 5.16 ms bf16.

The largest int4 quantized matmul is slower than dense bf16 despite moving fewer weight bytes. Full-weight dequantization alone is small relative to matmul time, so the cost is inside the fused quantized matmul path, not a separate per-step `mx.quantize` pass.

## Bottleneck Classification

Evidence-supported ranking:

1. Quantized linear execution inside transformer blocks, especially single-stream blocks.
2. Double-stream MLP and attention projection modules.
3. Elementwise residual/norm/modulation work.
4. Host graph construction, scheduler, command-buffer encoding/submission, RoPE, and layout conversions.
5. Pure SDPA attention kernels.

Bound classification:

- Not scheduler-bound: scheduler graph + sync is ~0.8 ms in the timeline.
- Not host graph-bound: actual transformer graph construction median is ~10.5 ms.
- Not command-submission-bound: Metal app-side encoder time is 27.7 ms over the warmed step.
- Not pure-attention-bound: previous `profile_step.py` sanity check measured pure SDPA x25 at 70.3 ms, 0.7% of the step.
- Not memory-capacity improving latency: bf16 uses much more memory and is faster than int4/int8.
- Actual DRAM bandwidth counters were not available; only lower-bound tensor bytes were measured. Lower-bound bandwidth for the largest layers is 1.5-2.8 GB/s, far below any plausible Apple Silicon memory-bandwidth ceiling, so the measured slowdown is not explained by required tensor traffic alone.

Metal occupancy/counter status:

- `Metal GPU Counters` recorded the warning: "Selected counter profile is not supported on target device."
- Exported GPU trace had 0 rows for `metal-shader-profiler-intervals` and `metal-gpu-intervals`.
- Shader list for the Python process did identify MLX kernels including `affine_qmm_t_bfloat16_t_gs_64_b_4_alN_true_batch_0`, `affine_dequantize_bfloat16_t_gs_64_b_4`, and `steel_attention_bfloat16_bq32_bk16_bd128...`.
- Register pressure, threadgroup utilization, and occupancy could not be measured from available profiler exports. Therefore occupancy is not proven as the limiter.

## Code References

- Per-load quantization happens in `mxalloy/models/flux2/loader.py:56` and `mxalloy/models/flux2/loader.py:77`.
- Single-stream fused QKV/MLP projection is defined at `mxalloy/models/flux2/transformer.py:269` and executed at `mxalloy/models/flux2/transformer.py:278`.
- Single-stream output projection is defined at `mxalloy/models/flux2/transformer.py:275` and executed at `mxalloy/models/flux2/transformer.py:303`.
- Double-stream attention projections are defined at `mxalloy/models/flux2/transformer.py:204`.
- Transformer loop and RoPE cache are in `mxalloy/models/flux2/transformer.py:478` and `mxalloy/models/flux2/transformer.py:498`.
- Scheduler step is a single Euler update in `mxalloy/models/flux2/scheduler.py:42`.

## Optimization Roadmap

1. Replace or improve MLX affine quantized matmul for the large projection shapes.
   Expected gain: high if the current 3.6-3.8 TFLOP/s can move toward 6-8 TFLOP/s; this is the only kernel-level path that can plausibly save multiple seconds.
   Difficulty: high.
   Confidence: medium, because profiler counters are missing but layer timing is decisive.

2. Hybrid bf16 for selected largest layers when memory allows.
   Expected gain: measured whole-step bf16 is 1.38 s faster than int4; selectively dequantizing the 20 single-stream projections would capture a meaningful part of that.
   Difficulty: medium.
   Confidence: high for speed, low for fitting comfortably on 18 GB without careful selection.

3. Fuse single-stream block internals around `to_qkv_mlp_proj`, split/reshape/norm/RoPE, SDPA, activation, concat, and `to_out`.
   Expected gain: potentially hundreds of milliseconds to ~1 s, but not enough alone for sub-5 s.
   Difficulty: high.
   Confidence: medium.

4. Structural/model-work reduction is required for sub-5 s.
   Expected gain: the measured bf16 upper bound is still 9.42 s, so kernel swaps alone need more than 2x improvement. Reducing layers/tokens/work or using a distilled variant is the realistic route to sub-5.
   Difficulty: high.
   Confidence: high.

Low-value areas:

- Attention-only optimization: pure SDPA is ~70 ms per step.
- RoPE recomputation/cache: ~3 ms in the measured timeline.
- Scheduler/host dispatch: below 1% combined.
