// Fused quantized-KV attention kernel (Metal) -- compiled metallib entry points.
//
// O = softmax(Q Kᵀ·scale) V with K,V in MLX affine int8/int4 group format, dequantized
// inline in registers (the fp16 K/V never materialise on the heap -- the memory win).
//
// v1 below is the *validated* one-thread-per-query body, ported verbatim from
// experiments/proto_qkv_attention.py (parity 4.9e-04 vs dequant->SDPA, 4- and 8-bit). It is
// correctness/memory-first, not speed-tuned (register-heavy). The speed pass replaces this
// body with MLX Steel attention: simdgroup_matrix MMA (steel/attn/mma.h) for QKᵀ and P·V +
// QuantizedBlockLoader (quantized.h) staging packed K/V into threadgroup memory, with
// simd_shuffle/simd_sum/simd_max for the online-softmax reductions. The dispatch/binding
// contract (buffers 0..6 = inputs, 7 = out, 8 = params) stays the same.

#include <metal_stdlib>

using namespace metal;

// Layout must match QSDPAParams in quantized_sdpa.cpp (all 4-byte members, no padding).
struct QSDPAParams {
  int B, H, L, S;
  int group_size;
  float scale;
};

// One thread -> one output row O[b,h,i,:]. T = q/o dtype; BITS = 8|4; D = head_dim.
template <typename T, int BITS, int D>
[[kernel]] void qsdpa(
    const device T* q [[buffer(0)]],     // (B,H,L,D)
    const device uint* kw [[buffer(1)]], // (B,H,S,D*BITS/32) packed
    const device T* ks [[buffer(2)]],    // (B,H,S,D/group)
    const device T* kb [[buffer(3)]],
    const device uint* vw [[buffer(4)]],
    const device T* vs [[buffer(5)]],
    const device T* vb [[buffer(6)]],
    device T* out [[buffer(7)]],         // (B,H,L,D)
    constant QSDPAParams& p [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= uint(p.B * p.H * p.L)) {
    return;
  }
  constexpr int VPW = 32 / BITS;          // packed values per uint32 word
  constexpr int NWORDS = D * BITS / 32;   // words per K/V row
  constexpr uint MASK = (1u << BITS) - 1u;
  const int G = p.group_size;

  const int i = int(gid % uint(p.L));
  const int h = int((gid / uint(p.L)) % uint(p.H));
  const int b = int(gid / uint(p.L * p.H));
  const uint q_base = ((uint(b) * p.H + h) * p.L + i) * D;
  const uint kv_bh = uint(b) * p.H + h;

  float qreg[D];
  for (int d = 0; d < D; ++d) {
    qreg[d] = float(q[q_base + d]);
  }

  float m = -1e30f, l = 0.0f;
  float acc[D];
  for (int d = 0; d < D; ++d) {
    acc[d] = 0.0f;
  }

  for (int s = 0; s < p.S; ++s) {
    const uint row = kv_bh * uint(p.S) + uint(s);
    const uint wbase = row * NWORDS;
    const uint gbase = row * uint(D / G);

    float score = 0.0f;  // inline-dequant K, score = q·k
    for (int d = 0; d < D; ++d) {
      const uint nib = (kw[wbase + d / VPW] >> (BITS * (d % VPW))) & MASK;
      const int g = d / G;
      score += qreg[d] * (float(nib) * float(ks[gbase + g]) + float(kb[gbase + g]));
    }
    score *= p.scale;

    const float mnew = max(m, score);  // online softmax
    const float corr = exp(m - mnew);
    const float pp = exp(score - mnew);
    l = l * corr + pp;

    for (int d = 0; d < D; ++d) {  // inline-dequant V, acc += p·v
      const uint nib = (vw[wbase + d / VPW] >> (BITS * (d % VPW))) & MASK;
      const int g = d / G;
      acc[d] = acc[d] * corr + pp * (float(nib) * float(vs[gbase + g]) + float(vb[gbase + g]));
    }
    m = mnew;
  }

  const float inv = 1.0f / l;
  for (int d = 0; d < D; ++d) {
    out[q_base + d] = T(acc[d] * inv);
  }
}

// Exported entry points; host_name must match the names eval_gpu builds. fp16 first
// (validated); bfloat + more head_dims come with the speed pass.
template [[host_name("mxalloy_qsdpa_float16_b8_d128")]] [[kernel]] void
qsdpa<half, 8, 128>(
    const device half*, const device uint*, const device half*, const device half*,
    const device uint*, const device half*, const device half*, device half*,
    constant QSDPAParams&, uint);
template [[host_name("mxalloy_qsdpa_float16_b4_d128")]] [[kernel]] void
qsdpa<half, 4, 128>(
    const device half*, const device uint*, const device half*, const device half*,
    const device uint*, const device half*, const device half*, device half*,
    constant QSDPAParams&, uint);
