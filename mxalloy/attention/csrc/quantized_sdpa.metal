// Fused quantized-KV flash-attention kernel (Metal) -- SKELETON.
//
// One GPU dispatch computes  O = softmax(Q Kᵀ · scale) V  with K and V read from MLX's
// affine int4/int8 group format and dequantized **inline in registers** -- the 16-bit K/V
// never touch the unified-memory heap. This is the memory win over `dequantize -> SDPA`.
//
// Algorithm: classic flash-attention with online softmax (Dao et al.), one threadgroup per
// (batch, head, query-tile):
//   load Q tile -> for each K/V key-block: inline-dequant K block, S = QKᵀ·scale,
//   update running max m and sum l, rescale accumulator O, inline-dequant V block, O += P·V.
//   write O. Row max/sum use SIMD-group reductions; blocks stream through threadgroup memory.
//
// Two implementations of this body exist:
//   * the simple one-thread-per-query version below -- correctness/memory-first, validated
//     end-to-end (4-bit AND 8-bit) in experiments/proto_qkv_attention.py via mx.fast.metal_kernel.
//   * the OPTIMIZED target (the speed win): don't hand-roll. MLX ships the pieces as device
//     headers -- build on <mlx/.../kernels/steel/attn/mma.h> (simdgroup_matrix MMA for the QKᵀ
//     and P·V matmuls) and swap `QuantizedBlockLoader` (<mlx/.../kernels/quantized.h>) in for
//     the K/V loads so packed blocks stage into *threadgroup* memory and dequant on the way
//     (spike stays off the global heap). simd_shuffle/simd_sum/simd_max carry the online-softmax
//     row reductions. Default precision is 8-bit (byte-aligned -> simpler/faster staging than
//     4-bit nibbles); 4-bit is the OOM-only lever.
// TODOs mark the parts to finish + verify on-device (needs the Metal toolchain / full Xcode).

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

// ---- inline affine dequant of one packed group (illustrative; 4-bit shown) --------------
// MLX packs `bits`-wide values little-endian into uint32 words; per group of `group_size`
// values there is one (scale, bias): value = q * scale + bias.
template <typename T, int bits>
inline T dequant_one(const device uint32_t* w, uint idx, T scale, T bias) {
  constexpr uint vals_per_word = 32 / bits;
  constexpr uint mask = (1u << bits) - 1u;
  uint32_t word = w[idx / vals_per_word];
  uint shift = (idx % vals_per_word) * bits;
  uint q = (word >> shift) & mask;
  return static_cast<T>(q) * scale + bias;  // TODO: 3/5/6-bit are not word-aligned; use MLX helper
}

// ---- params (set via set_bytes from eval_gpu) ------------------------------------------
struct QSDPAParams {
  int B, H, L, S, D;   // batch, heads, query len, key len, head_dim
  int group_size;      // quant group along D
  float scale;         // softmax scale (1/sqrt(D))
  int has_mask;        // 0/1
  int q_tile, k_tile;  // tiling
};

// One threadgroup -> one (batch, head, query-tile). T is the q/o dtype (half / bfloat).
template <typename T, int BITS, int D>
[[kernel]] void qsdpa_kernel(
    const device T* q          [[buffer(0)]],   // (B,H,L,D)
    const device uint32_t* k_w [[buffer(1)]],   // (B,H,S,D*BITS/32)
    const device T* k_s        [[buffer(2)]],   // (B,H,S,D/group)
    const device T* k_b        [[buffer(3)]],
    const device uint32_t* v_w [[buffer(4)]],
    const device T* v_s        [[buffer(5)]],
    const device T* v_b        [[buffer(6)]],
    device T* out              [[buffer(7)]],   // (B,H,L,D)
    constant QSDPAParams& p    [[buffer(8)]],
    uint3 tgid                 [[threadgroup_position_in_grid]],
    uint3 lid                  [[thread_position_in_threadgroup]],
    uint simd_lane             [[thread_index_in_simdgroup]],
    uint simd_group            [[simdgroup_index_in_threadgroup]]) {
  // --- locate this threadgroup's (b, h, query rows) ---
  const int b = tgid.z / p.H;
  const int h = tgid.z % p.H;
  const int q0 = tgid.y * p.q_tile;  // first query row this TG owns
  // TODO: bounds, GQA head remap (k/v head = h / (H/n_kv_heads)).

  // --- per-query online-softmax state held in registers ---
  T acc[D];                  // O accumulator for this thread's query row
  for (int d = 0; d < D; ++d) acc[d] = T(0);
  float m = -INFINITY;       // running max of scores
  float l = 0.0f;            // running sum of exp(scores - m)

  threadgroup T k_block[/*k_tile*/ 128][D];  // dequantized K block (TODO: size from p.k_tile)
  threadgroup T v_block[/*k_tile*/ 128][D];

  // --- stream over key blocks ---
  for (int s0 = 0; s0 < p.S; s0 += p.k_tile) {
    // (1) cooperatively inline-dequant this K/V block into threadgroup memory.
    //     Each thread unpacks a slice; group scale/bias indexed by (col / group_size).
    //     TODO: coalesce loads; reuse MLX QuantizedBlockLoader for the optimized path.
    // for (col owned by this thread) {
    //   T sc = k_s[...], bs = k_b[...];
    //   k_block[r][col] = dequant_one<T,BITS>(k_w + base, col, sc, bs);
    //   ... same for v_block ...
    // }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // (2) scores S = Q·Kᵀ·scale for the keys in this block, online-softmax update.
    for (int r = 0; r < p.k_tile && (s0 + r) < p.S; ++r) {
      float s = 0.0f;
      for (int d = 0; d < D; ++d) s += float(q[/*q row*/ 0 * D + d]) * float(k_block[r][d]);
      s *= p.scale;
      if (p.has_mask) { /* TODO apply additive/bool/causal mask */ }
      float m_new = max(m, s);
      float corr = exp(m - m_new);     // rescale prior accumulator
      float pw = exp(s - m_new);       // weight of this key
      l = l * corr + pw;
      for (int d = 0; d < D; ++d) acc[d] = T(float(acc[d]) * corr + pw * float(v_block[r][d]));
      m = m_new;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // (3) normalize and write. (Row max/sum can use simd_shuffle reductions when a row is
  //     split across lanes; the per-thread-per-row layout above keeps it local for clarity.)
  const float inv_l = 1.0f / l;
  for (int d = 0; d < D; ++d) out[/*o row*/ 0 * D + d] = T(float(acc[d]) * inv_l);
  (void)q0; (void)b; (void)h; (void)simd_lane; (void)simd_group; (void)lid;
}

// ---- explicit instantiations the metallib exports (host_name must match eval_gpu) -------
// MLX's metallib build compiles these named entry points; eval_gpu looks them up by name.
template [[host_name("mxalloy_qsdpa_float16_b4_d128")]] [[kernel]] void
qsdpa_kernel<half, 4, 128>(
    const device half*, const device uint32_t*, const device half*, const device half*,
    const device uint32_t*, const device half*, const device half*, device half*,
    constant QSDPAParams&, uint3, uint3, uint, uint);
template [[host_name("mxalloy_qsdpa_bfloat16_b4_d128")]] [[kernel]] void
qsdpa_kernel<bfloat, 4, 128>(
    const device bfloat*, const device uint32_t*, const device bfloat*, const device bfloat*,
    const device uint32_t*, const device bfloat*, const device bfloat*, device bfloat*,
    constant QSDPAParams&, uint3, uint3, uint, uint);
// TODO: add b8 (int8) and other head_dims (64/96/256) as supported configs grow.
