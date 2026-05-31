"""Performance-ceiling analysis for klein-4B per-step time on Apple Silicon.

Times every klein projection shape at its real token count in int4/int8/bf16, builds the
full per-step breakdown (which layers dominate), and runs roofline math (effective TFLOP/s
vs GB/s) to classify the quantized path. Answers: hybrid (selective bf16) viability + the
realistic step-latency floor.

    PYTHONPATH=. .venv/bin/python experiments/perf_ceiling.py
"""

from __future__ import annotations

import statistics
import time

import mlx.core as mx

# klein-4B projection inventory: (name, out, in, seq_tokens, count_per_step)
# image stream = 4096 tokens (1024**2 packed latent), text stream = 512, single blocks = 4608.
INV = [
    ("double img attn (q,k,v,out)", 3072, 3072, 4096, 5 * 4),
    ("double txt attn (q,k,v,out)", 3072, 3072, 512, 5 * 4),
    ("double img ff_in", 18432, 3072, 4096, 5),
    ("double img ff_out", 3072, 9216, 4096, 5),
    ("double txt ff_in", 18432, 3072, 512, 5),
    ("double txt ff_out", 3072, 9216, 512, 5),
    ("single qkv_mlp", 27648, 3072, 4608, 20),
    ("single to_out", 3072, 12288, 4608, 20),
    ("context_embedder", 3072, 7680, 512, 1),
    ("proj_out", 128, 3072, 4096, 1),
]


def bench(fn, iters=20, warmup=6) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t = time.perf_counter(); mx.eval(fn()); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3  # ms


def time_shape(out_d, in_d, seq):
    x = mx.random.normal((seq, in_d)).astype(mx.bfloat16)
    w = mx.random.normal((out_d, in_d)).astype(mx.bfloat16)
    wt = w.T
    res = {}
    res["bf16"] = bench(lambda: mx.matmul(x, wt))
    for bits in (8, 4):
        wq, s, b = mx.quantize(w, group_size=64, bits=bits)
        res[f"int{bits}"] = bench(
            lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=bits)
        )
    return res


def main() -> None:
    try:
        di = mx.device_info()
        print("device:", {k: di[k] for k in di if "name" in k.lower() or "memory" in k.lower() or "recommend" in k.lower()})
    except Exception as e:  # noqa: BLE001
        print("device_info:", e)
    print()

    totals = {"bf16": 0.0, "int8": 0.0, "int4": 0.0}
    rows = []
    for name, out_d, in_d, seq, count in INV:
        r = time_shape(out_d, in_d, seq)
        flops = 2 * seq * in_d * out_d
        tflops_bf16 = flops / (r["bf16"] / 1e3) / 1e12
        for dt in totals:
            totals[dt] += r[dt] * count
        rows.append((name, count, r, flops, tflops_bf16))
        print(f"{name:32s} x{count:<3d} seq{seq:<5d} "
              f"bf16 {r['bf16']:6.2f}  int8 {r['int8']:6.2f}  int4 {r['int4']:6.2f} ms  "
              f"| int4/bf16 {r['int4']/r['bf16']:.2f}x  bf16 {tflops_bf16:4.1f} TFLOP/s")

    print("\n--- per-step totals (sum over all projections) ---")
    for dt in ("int4", "int8", "bf16"):
        print(f"  {dt}: {totals[dt]/1e3:6.3f} s/step")
    print(f"  int4/bf16 step ratio: {totals['int4']/totals['bf16']:.2f}x")

    print("\n--- top contributors to int4 step time ---")
    contrib = sorted(rows, key=lambda r: r[2]["int4"] * r[1], reverse=True)
    for name, count, r, flops, _ in contrib[:6]:
        print(f"  {name:32s} {r['int4']*count/1e3:5.3f} s  ({r['int4']*count/totals['int4']*100:4.1f}%)")

    print("\n--- hybrid: big projections (>=18432 wide) bf16, rest int4 ---")
    hyb = 0.0
    bf16_mem_add = 0.0
    for name, count, r, flops, _ in rows:
        out_d = [x for x in INV if x[0] == name][0][1]
        in_d = [x for x in INV if x[0] == name][0][2]
        big = max(out_d, in_d) >= 18000
        hyb += (r["bf16"] if big else r["int4"]) * count
        if big:
            bf16_mem_add += (2 - 0.5) * in_d * out_d * count / 1e9  # bf16 vs int4 weight bytes
    print(f"  hybrid step: {hyb/1e3:.3f} s  (vs int4 {totals['int4']/1e3:.3f}, bf16 {totals['bf16']/1e3:.3f})")
    print(f"  extra resident weight memory for the bf16 big-projections: +{bf16_mem_add:.2f} GB")


if __name__ == "__main__":
    main()
