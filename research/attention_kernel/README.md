# Fused quantized-KV attention (experimental, frozen)

A custom MLX `Primitive` (`_ext.so` + `mxalloy_ext.metallib`) computing
`O = softmax(QKᵀ·scale)·V` with **K and V supplied in MLX affine-quant group format**,
dequantized inline in-register so the 16-bit K/V never materialise on the unified-memory heap.

## Status: frozen

This is **research**, not part of the shipped package. It is correct — verified against the
pure-MLX oracle in `mxalloy/attention/quantized_sdpa.py` to ~5e-04 — but on the current
GEMM-bound FLUX diffusion path it is *memory-not-speed*: attention is ~0.7% of a denoise step,
and MLX's Steel SDPA is 3–6× faster on dense Q. The win is real only where attention is the
memory/latency cost: **autoregressive decode, long context, KV-cached / continuous-batching**
serving. mxalloy therefore ships the pure-MLX fallback as the live primitive and keeps this
here for the day a KV-cache workload (e.g. an LLM path) makes it pay.

## What's here

- `quantized_sdpa.metal` — the flash-attention kernel: multi-simdgroup, `simdgroup_matrix`
  half-MMA / float-accumulate, online softmax, inline int4/int8 group dequant of K/V.
- `quantized_sdpa.{h,cpp}` — the `Primitive` subclass + `eval_gpu` dispatch.
- `bindings.cpp` — nanobind binding (`NB_DOMAIN mlx` + `STABLE_ABI`, to match MLX's ABI so the
  op accepts `mlx.core.array` directly).
- `CMakeLists.txt`, `setup.py` — the build.

## Build (optional, to A/B)

Requires full Xcode + the Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`):

```bash
pip install nanobind cmake
python research/attention_kernel/setup.py build_ext --inplace
# then drop the built _ext*.so + mxalloy_ext.metallib next to
# mxalloy/attention/quantized_sdpa.py
```

The Python surface (`mxalloy.attention.quantized_scaled_dot_product_attention`) auto-detects a
built `_ext` and uses it; otherwise it transparently uses the pure-MLX fallback.
