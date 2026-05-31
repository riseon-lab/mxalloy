"""Stage-1 prototype: fused quantized-KV attention via mx.fast.metal_kernel (runs today).

Validates the *exact kernel body* destined for the compiled metallib, without Xcode:
inline-dequant int4 K/V in-register + online-softmax, in one dispatch. Two checks:
  1. parity  -- fused kernel vs the dequant->SDPA fallback (same 4-bit K/V) and vs full-prec.
  2. peak    -- fused (no transient fp16 K/V) vs fallback (materialises fp16 K/V), on 18GB.

v1 scope (restraint): MHA, fp16 q, 4-bit K/V, head_dim 128, no mask, one thread per query
(correctness + memory first; tiling/SIMD-matmul is the metallib stage). The body ports
verbatim into research/attention_kernel/quantized_sdpa.metal.

    PYTHONPATH=. .venv/bin/python experiments/proto_qkv_attention.py
"""

from __future__ import annotations

import mlx.core as mx

from mxalloy.attention.quantized_sdpa import _dequant, quantize_kv

_HEADER = """
#define HEAD_DIM 128
#define GROUP {group}
#define BITS {bits}
#define VPW (32 / BITS)
#define NWORDS (HEAD_DIM * BITS / 32)
#define NGROUPS (HEAD_DIM / GROUP)
"""

_SRC = r"""
  uint gid = thread_position_in_grid.x;
  int B = dims[0]; int H = dims[1]; int L = dims[2]; int S = dims[3];
  if (gid >= (uint)(B * H * L)) return;
  int i = (int)(gid % (uint)L);
  int h = (int)((gid / (uint)L) % (uint)H);
  int b = (int)(gid / (uint)(L * H));
  float sc = scale[0];

  uint q_base = ((((uint)b * H + h) * L) + i) * HEAD_DIM;
  uint kv_bh  = ((uint)b * H + h);

  float qreg[HEAD_DIM];
  for (int d = 0; d < HEAD_DIM; ++d) qreg[d] = (float)q[q_base + d];

  float m = -1e30f, l = 0.0f;
  float acc[HEAD_DIM];
  for (int d = 0; d < HEAD_DIM; ++d) acc[d] = 0.0f;

  for (int s = 0; s < S; ++s) {
    uint row = kv_bh * (uint)S + (uint)s;
    uint w_base = row * NWORDS, g_base = row * NGROUPS;

    float score = 0.0f;                       // inline-dequant K, score = q.k
    for (int d = 0; d < HEAD_DIM; ++d) {
      uint word = kw[w_base + d / VPW];
      uint nib  = (word >> (BITS * (d % VPW))) & ((1u << BITS) - 1u);
      int g = d / GROUP;
      score += qreg[d] * ((float)nib * (float)ks[g_base + g] + (float)kb[g_base + g]);
    }
    score *= sc;

    float mnew = max(m, score);               // online softmax
    float corr = exp(m - mnew);
    float p    = exp(score - mnew);
    l = l * corr + p;

    for (int d = 0; d < HEAD_DIM; ++d) {       // inline-dequant V, acc += p * v
      uint word = vw[w_base + d / VPW];
      uint nib  = (word >> (BITS * (d % VPW))) & ((1u << BITS) - 1u);
      int g = d / GROUP;
      acc[d] = acc[d] * corr + p * ((float)nib * (float)vs[g_base + g] + (float)vb[g_base + g]);
    }
    m = mnew;
  }

  float inv = 1.0f / l;
  for (int d = 0; d < HEAD_DIM; ++d) out[q_base + d] = (half)(acc[d] * inv);
"""


def build_kernel(group: int, bits: int):
    # 4/8-bit share one body: VPW values per uint32, mask=(1<<bits)-1, x = q*scale + bias.
    return mx.fast.metal_kernel(
        name=f"qkv_attn_b{bits}_g{group}",
        input_names=["q", "kw", "ks", "kb", "vw", "vs", "vb", "dims", "scale"],
        output_names=["out"],
        header=_HEADER.format(group=group, bits=bits),
        source=_SRC,
    )


def fused(kern, q, k, v, scale):
    b, h, length, d = q.shape
    s = k.weights.shape[-2]
    dims = mx.array([b, h, length, s, d, k.group_size], dtype=mx.int32)
    (out,) = kern(
        inputs=[q, k.weights, k.scales, k.biases, v.weights, v.scales, v.biases, dims,
                mx.array([scale], dtype=mx.float32)],
        grid=(b * h * length, 1, 1),
        threadgroup=(min(256, b * h * length), 1, 1),
        output_shapes=[(b, h, length, d)],
        output_dtypes=[q.dtype],
    )
    return out


def gb(n: int) -> float:
    return round(n / 1e9, 3)


def main() -> None:
    mx.random.seed(0)

    # --- parity (small, self-attention) across the precisions we care about ---
    b, h, length, d = 1, 4, 96, 128
    q = mx.random.normal((b, h, length, d)).astype(mx.float16)
    k = mx.random.normal((b, h, length, d)).astype(mx.float16)
    v = mx.random.normal((b, h, length, d)).astype(mx.float16)
    scale = 1.0 / (d**0.5)
    full = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    for bits, gs in [(8, 64), (4, 64)]:
        kern = build_kernel(group=gs, bits=bits)
        kq, vq = quantize_kv(k, gs, bits), quantize_kv(v, gs, bits)
        out = fused(kern, q, kq, vq, scale)
        fb = mx.fast.scaled_dot_product_attention(q, _dequant(kq), _dequant(vq), scale=scale)
        mx.eval(out, fb)
        vs_fb = float(mx.max(mx.abs(out - fb)))
        vs_full = float(mx.max(mx.abs(out - full)))
        print(f"{bits}-bit gs{gs}: parity vs fallback {vs_fb:.2e}  | vs full-precision {vs_full:.2e}")

    # --- peak: long context, fused vs dequant->SDPA, at the 8-bit target ---
    bits, gs = 8, 64
    kern = build_kernel(group=gs, bits=bits)
    b, h, length, s, d = 1, 32, 128, 16384, 128
    q = mx.random.normal((b, h, length, d)).astype(mx.float16)
    k = mx.random.normal((b, h, s, d)).astype(mx.float16)
    v = mx.random.normal((b, h, s, d)).astype(mx.float16)
    kq, vq = quantize_kv(k, gs, bits), quantize_kv(v, gs, bits)
    del k, v
    mx.eval(q, kq.weights, kq.scales, kq.biases, vq.weights, vq.scales, vq.biases)
    fp16_kv = gb(2 * b * h * s * d * 2)  # the transient the fallback materialises

    mx.clear_cache(); mx.reset_peak_memory()
    of = fused(kern, q, kq, vq, scale); mx.eval(of)
    peak_fused = gb(mx.get_peak_memory())

    mx.clear_cache(); mx.reset_peak_memory()
    od = mx.fast.scaled_dot_product_attention(q, _dequant(kq), _dequant(vq), scale=scale)
    mx.eval(od)
    peak_fb = gb(mx.get_peak_memory())

    print(f"\n8-bit long-context {h}h x {s} keys (D={d}):  fp16 K/V transient = {fp16_kv} GB")
    print(f"peak fused (inline dequant): {peak_fused} GB")
    print(f"peak dequant->SDPA fallback: {peak_fb} GB")
    print(f"peak saved by fusing:        {round(peak_fb - peak_fused, 3)} GB")


if __name__ == "__main__":
    main()
