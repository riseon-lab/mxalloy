"""Precision microbench: which precision is fastest for klein-scale projection GEMMs here.

Answers the crux of the adaptive-execution question: at diffusion sequence lengths (~4096
image tokens), is int4/int8 ``quantized_matmul`` faster or slower than bf16/fp16 ``matmul`` on
this Apple Silicon machine? Quantization is a *memory* lever; this isolates its *speed* cost so
the execution strategy can trade memory<->speed against real numbers instead of assertion.

Representative klein-4B inner GEMMs (inner dim 3072, MLP 12288), at M=4096 tokens (1024^2).
Not model-specific machinery -- just the four GEMM shapes that dominate a denoise step.

    PYTHONPATH=. .venv/bin/python experiments/precision_microbench.py
"""

from __future__ import annotations

import time

import mlx.core as mx

M = 4096  # image tokens at 1024^2 (64x64 patches)
GROUP = 64
ITERS = 30
WARMUP = 8

GEMMS = [
    ("attn_qkv   3072->9216", 3072, 9216),
    ("attn_out   3072->3072", 3072, 3072),
    ("mlp_up    3072->12288", 3072, 12288),
    ("mlp_down  12288->3072", 12288, 3072),
]


def bench(make) -> float:
    for _ in range(WARMUP):
        mx.eval(make())
    t0 = time.perf_counter()
    for _ in range(ITERS):
        mx.eval(make())
    return (time.perf_counter() - t0) / ITERS * 1e3  # ms / iter


def run(name: str, in_f: int, out_f: int) -> None:
    flops = 2 * M * in_f * out_f
    x = mx.random.normal((M, in_f)).astype(mx.bfloat16)
    w = mx.random.normal((out_f, in_f)).astype(mx.bfloat16)

    res = {"bf16": bench(lambda: x @ w.T)}
    xf, wf = x.astype(mx.float16), w.astype(mx.float16)
    res["fp16"] = bench(lambda: xf @ wf.T)
    for bits in (8, 4):
        wq, sc, bi = mx.quantize(w, group_size=GROUP, bits=bits)
        res[f"int{bits}"] = bench(
            lambda wq=wq, sc=sc, bi=bi, bits=bits: mx.quantized_matmul(
                x, wq, sc, bi, transpose=True, group_size=GROUP, bits=bits
            )
        )

    base = res["bf16"]
    print(f"\n{name}   (M={M}, {flops / 1e9:.1f} GFLOP, weight {out_f * in_f / 1e6:.1f}M params)")
    for k in ("bf16", "fp16", "int8", "int4"):
        ms = res[k]
        print(f"  {k:5s} {ms:7.3f} ms   {flops / 1e9 / ms:6.2f} TFLOP/s   {ms / base:5.2f}x vs bf16")


def main() -> None:
    print(f"device: {mx.default_device()}")
    for name, i, o in GEMMS:
        run(name, i, o)
    print(
        "\nWeight bytes/param: bf16/fp16 2.0 | int8 ~1.06 (8b + scales) | int4 ~0.56 (4b + scales)"
    )


if __name__ == "__main__":
    main()
