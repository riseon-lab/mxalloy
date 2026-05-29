// Fused quantized-KV attention kernel (Metal) -- simdgroup-matrix (MMA) flash, v1.
//
// O = softmax(Q Kᵀ·scale) V with K,V in MLX affine int8/int4 group format, dequantized
// per-tile into threadgroup memory (the fp16 K/V never hit the global heap).
//
// v1 structure (correctness/occupancy-simple): one simdgroup (32 threads) per
// (batch, head, 8-query block). MMA (simdgroup_matrix 8x8) does QKᵀ and P·V; the online
// softmax + the O accumulator live in threadgroup memory, where per-row rescaling is
// trivial. Tiles stage as float for a clean first cut.
// v2 (speed): QuantizedBlockLoader for coalesced dequant-staging, half-input MMA with
// float accumulate, and multiple simdgroups per threadgroup for occupancy.

#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

// Layout must match QSDPAParams in quantized_sdpa.cpp.
struct QSDPAParams {
  int B, H, L, S;
  int group_size;
  float scale;
};

template <typename T, int BITS, int D>
[[kernel]] void qsdpa(
    const device T* q [[buffer(0)]],
    const device uint* kw [[buffer(1)]],
    const device T* ks [[buffer(2)]],
    const device T* kb [[buffer(3)]],
    const device uint* vw [[buffer(4)]],
    const device T* vs [[buffer(5)]],
    const device T* vb [[buffer(6)]],
    device T* out [[buffer(7)]],
    constant QSDPAParams& p [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int BQ = 8, BK = 8;
  constexpr int VPW = 32 / BITS;
  constexpr int NWORDS = D * BITS / 32;
  constexpr int DCH = D / 8;
  constexpr uint MASK = (1u << BITS) - 1u;
  const int G = p.group_size;
  const int NGROUPS = D / G;

  const int q0 = int(tgid.x) * BQ;  // first query row of this block
  const uint bh = tgid.z;           // b*H + h
  const int nq = min(BQ, p.L - q0);
  if (q0 >= p.L) {
    return;
  }

  threadgroup float Qs[BQ][D];
  threadgroup float Ks[BK][D];
  threadgroup float Vs[BK][D];
  threadgroup float Ps[BQ][BK];
  threadgroup float Ss[BQ][BK];
  threadgroup float Os[BQ][D];
  threadgroup float dO[BQ][D];
  threadgroup float m_[BQ], l_[BQ];

  for (int idx = int(lane); idx < BQ * D; idx += 32) {
    int r = idx / D, c = idx % D;
    Qs[r][c] = (r < nq) ? float(q[(bh * p.L + (q0 + r)) * D + c]) : 0.0f;
    Os[r][c] = 0.0f;
  }
  for (int i = int(lane); i < BQ; i += 32) {
    m_[i] = -1e30f;
    l_[i] = 0.0f;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  for (int k0 = 0; k0 < p.S; k0 += BK) {
    int nk = min(BK, p.S - k0);

    // dequant K/V tiles (BK x D) into threadgroup (spike-free); zero-pad keys >= nk.
    for (int idx = int(lane); idx < BK * D; idx += 32) {
      int r = idx / D, c = idx % D;
      if (r < nk) {
        uint row = bh * p.S + uint(k0 + r);
        uint wb = row * NWORDS, gb = row * uint(NGROUPS);
        uint kn = (kw[wb + c / VPW] >> (BITS * (c % VPW))) & MASK;
        uint vn = (vw[wb + c / VPW] >> (BITS * (c % VPW))) & MASK;
        Ks[r][c] = float(kn) * float(ks[gb + c / G]) + float(kb[gb + c / G]);
        Vs[r][c] = float(vn) * float(vs[gb + c / G]) + float(vb[gb + c / G]);
      } else {
        Ks[r][c] = 0.0f;
        Vs[r][c] = 0.0f;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // S(BQ x BK) = Qs(BQ x D) * Ks^T(D x BK)
    simdgroup_matrix<float, 8, 8> Sacc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int dc = 0; dc < DCH; ++dc) {
      simdgroup_matrix<float, 8, 8> Qf, Kf;
      simdgroup_load(Qf, &Qs[0][dc * 8], D);
      simdgroup_load(Kf, &Ks[0][dc * 8], D, ulong2(0, 0), /*transpose=*/true);
      simdgroup_multiply_accumulate(Sacc, Qf, Kf, Sacc);
    }
    simdgroup_store(Sacc, &Ss[0][0], BK);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // online softmax: lanes 0..BQ-1 each own one query row.
    if (lane < uint(BQ)) {
      int i = int(lane);
      float rmax = -1e30f;
      for (int j = 0; j < nk; ++j) {
        rmax = max(rmax, Ss[i][j] * p.scale);
      }
      float mnew = max(m_[i], rmax);
      float corr = exp(m_[i] - mnew);
      float lsum = 0.0f;
      for (int j = 0; j < BK; ++j) {
        float pj = (j < nk) ? exp(Ss[i][j] * p.scale - mnew) : 0.0f;
        Ps[i][j] = pj;
        lsum += pj;
      }
      l_[i] = l_[i] * corr + lsum;
      m_[i] = mnew;
      for (int d = 0; d < D; ++d) {
        Os[i][d] *= corr;  // rescale running accumulator before adding this tile
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // dO(BQ x D) = Ps(BQ x BK) * Vs(BK x D); accumulate into Os.
    for (int dc = 0; dc < DCH; ++dc) {
      simdgroup_matrix<float, 8, 8> Oacc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      simdgroup_matrix<float, 8, 8> Pf, Vf;
      simdgroup_load(Pf, &Ps[0][0], BK);
      simdgroup_load(Vf, &Vs[0][dc * 8], D);
      simdgroup_multiply_accumulate(Oacc, Pf, Vf, Oacc);
      simdgroup_store(Oacc, &dO[0][dc * 8], D);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = int(lane); idx < BQ * D; idx += 32) {
      Os[idx / D][idx % D] += dO[idx / D][idx % D];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  // normalize + write
  if (lane < uint(nq)) {
    int i = int(lane);
    float inv = 1.0f / l_[i];
    for (int d = 0; d < D; ++d) {
      out[(bh * p.L + uint(q0 + i)) * D + d] = T(Os[i][d] * inv);
    }
  }
}

template [[host_name("mxalloy_qsdpa_float16_b8_d128")]] [[kernel]] void
qsdpa<half, 8, 128>(
    const device half*, const device uint*, const device half*, const device half*,
    const device uint*, const device half*, const device half*, device half*,
    constant QSDPAParams&, uint3, uint);
template [[host_name("mxalloy_qsdpa_float16_b4_d128")]] [[kernel]] void
qsdpa<half, 4, 128>(
    const device half*, const device uint*, const device half*, const device half*,
    const device uint*, const device half*, const device half*, device half*,
    constant QSDPAParams&, uint3, uint);
