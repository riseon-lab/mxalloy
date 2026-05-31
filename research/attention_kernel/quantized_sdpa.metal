// Fused quantized-KV attention kernel (Metal) -- simdgroup-matrix (MMA) flash, v2.
//
// O = softmax(Q Kᵀ·scale) V with K,V in MLX affine int8/int4 group format, dequantized
// per-tile into threadgroup memory (the fp16 K/V never hit the global heap).
//
// v2 over v1: NSG simdgroups per threadgroup *share one* dequantized K/V tile -- the
// dequant is paid once per NSG query-blocks (not once each) and occupancy is NSG x higher.
// Tiles stage as half (half-input MMA, float accumulate -> precision kept); the O
// accumulator + online softmax live in threadgroup memory (per-row rescale is trivial;
// P·V does load -> MMA-accumulate -> store, so no separate deltaO buffer).
// Next (v3): QuantizedBlockLoader coalesced staging, larger BK, dtype/GQA/mask breadth.

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
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tib [[thread_index_in_threadgroup]]) {
  constexpr int NSG = 4, BQ = 8, BK = 8;  // NSG must match the host dispatch (32*NSG threads)
  constexpr int VPW = 32 / BITS;
  constexpr int NWORDS = D * BITS / 32;
  constexpr int DCH = D / 8;
  constexpr uint MASK = (1u << BITS) - 1u;
  const int G = p.group_size;
  const int NGROUPS = D / G;

  const uint bh = tgid.z;                            // b*H + h
  const int tg_q0 = int(tgid.x) * (NSG * BQ);        // first query row of the threadgroup
  const int q0 = tg_q0 + int(sg) * BQ;               // first query row of this simdgroup
  const int nq = min(BQ, p.L - q0);                  // valid query rows (<=0 if past the end)

  threadgroup half Ks[BK][D];  // shared across simdgroups
  threadgroup half Vs[BK][D];
  threadgroup half Qs[NSG][BQ][D];
  threadgroup half Ps[NSG][BQ][BK];
  threadgroup float Ss[NSG][BQ][BK];
  threadgroup float Os[NSG][BQ][D];
  threadgroup float m_[NSG][BQ], l_[NSG][BQ];

  // load this simdgroup's Q tile (zero-pad rows past L); init its accumulator state.
  for (int idx = int(lane); idx < BQ * D; idx += 32) {
    int r = idx / D, c = idx % D;
    Qs[sg][r][c] = (r < nq) ? half(q[(bh * p.L + uint(q0 + r)) * D + c]) : half(0);
    Os[sg][r][c] = 0.0f;
  }
  if (lane < uint(BQ)) {
    m_[sg][lane] = -1e30f;
    l_[sg][lane] = 0.0f;
  }

  for (int k0 = 0; k0 < p.S; k0 += BK) {
    int nk = min(BK, p.S - k0);

    // all NSG*32 threads cooperatively dequant ONE K/V tile into shared threadgroup memory.
    for (int idx = int(tib); idx < BK * D; idx += NSG * 32) {
      int r = idx / D, c = idx % D;
      if (r < nk) {
        uint row = bh * p.S + uint(k0 + r);
        uint wb = row * NWORDS, gb = row * uint(NGROUPS);
        uint kn = (kw[wb + c / VPW] >> (BITS * (c % VPW))) & MASK;
        uint vn = (vw[wb + c / VPW] >> (BITS * (c % VPW))) & MASK;
        Ks[r][c] = half(float(kn) * float(ks[gb + c / G]) + float(kb[gb + c / G]));
        Vs[r][c] = half(float(vn) * float(vs[gb + c / G]) + float(vb[gb + c / G]));
      } else {
        Ks[r][c] = half(0);
        Vs[r][c] = half(0);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // S(BQ x BK) = Qs(BQ x D) * Ks^T(D x BK), half inputs -> float accumulate.
    simdgroup_matrix<float, 8, 8> Sacc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int dc = 0; dc < DCH; ++dc) {
      simdgroup_matrix<half, 8, 8> Qf, Kf;
      simdgroup_load(Qf, &Qs[sg][0][dc * 8], D);
      simdgroup_load(Kf, &Ks[0][dc * 8], D, ulong2(0, 0), /*transpose=*/true);
      simdgroup_multiply_accumulate(Sacc, Qf, Kf, Sacc);
    }
    simdgroup_store(Sacc, &Ss[sg][0][0], BK);
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // online softmax: this simdgroup's lanes 0..BQ-1 each own one query row.
    if (lane < uint(BQ)) {
      int i = int(lane);
      float rmax = -1e30f;
      for (int j = 0; j < nk; ++j) {
        rmax = max(rmax, Ss[sg][i][j] * p.scale);
      }
      float mnew = max(m_[sg][i], rmax);
      float corr = exp(m_[sg][i] - mnew);
      float lsum = 0.0f;
      for (int j = 0; j < BK; ++j) {
        float pj = (j < nk) ? exp(Ss[sg][i][j] * p.scale - mnew) : 0.0f;
        Ps[sg][i][j] = half(pj);
        lsum += pj;
      }
      l_[sg][i] = l_[sg][i] * corr + lsum;
      m_[sg][i] = mnew;
      for (int d = 0; d < D; ++d) {
        Os[sg][i][d] *= corr;  // rescale running accumulator before adding this tile
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Os += Ps(BQ x BK) * Vs(BK x D): load current chunk -> MMA-accumulate -> store back.
    for (int dc = 0; dc < DCH; ++dc) {
      simdgroup_matrix<float, 8, 8> Oacc;
      simdgroup_load(Oacc, &Os[sg][0][dc * 8], D);
      simdgroup_matrix<half, 8, 8> Pf, Vf;
      simdgroup_load(Pf, &Ps[sg][0][0], BK);
      simdgroup_load(Vf, &Vs[0][dc * 8], D);
      simdgroup_multiply_accumulate(Oacc, Pf, Vf, Oacc);
      simdgroup_store(Oacc, &Os[sg][0][dc * 8], D);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // all simdgroups done with this K/V tile
  }

  // normalize + write this simdgroup's rows. Signed compare: out-of-range simdgroups have
  // nq <= 0 (uint(nq) would wrap huge and write out of bounds).
  if (nq > 0 && int(lane) < nq) {
    int i = int(lane);
    float inv = 1.0f / l_[sg][i];
    for (int d = 0; d < D; ++d) {
      out[(bh * p.L + uint(q0 + i)) * D + d] = T(Os[sg][i][d] * inv);
    }
  }
}

template [[host_name("mxalloy_qsdpa_float16_b8_d128")]] [[kernel]] void
qsdpa<half, 8, 128>(
    const device half*, const device uint*, const device half*, const device half*,
    const device uint*, const device half*, const device half*, device half*,
    constant QSDPAParams&, uint3, uint, uint, uint);
template [[host_name("mxalloy_qsdpa_float16_b4_d128")]] [[kernel]] void
qsdpa<half, 4, 128>(
    const device half*, const device uint*, const device half*, const device half*,
    const device uint*, const device half*, const device half*, device half*,
    constant QSDPAParams&, uint3, uint, uint, uint);
