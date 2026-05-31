// nanobind module for the fused quantized-KV attention op -- SKELETON.
//
// The `mlx::core::array` <-> Python caster is provided by MLX's own bindings; an extension
// gets it by being built against the same nanobind + mlx (see mlx/examples/extensions).

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>

#include "quantized_sdpa.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mlx::core;

NB_MODULE(_ext, m) {
  m.doc() = "mxalloy fused quantized-KV attention (Metal extension)";

  m.def(
      "quantized_scaled_dot_product_attention",
      [](const array& q,
         const array& k_w,
         const array& k_s,
         const array& k_b,
         const array& v_w,
         const array& v_s,
         const array& v_b,
         float scale,
         int group_size,
         int bits,
         const std::optional<array>& mask) {
        return mxalloy::ext::quantized_scaled_dot_product_attention(
            q, k_w, k_s, k_b, v_w, v_s, v_b, scale, group_size, bits, mask);
      },
      "q"_a,
      "k_w"_a,
      "k_s"_a,
      "k_b"_a,
      "v_w"_a,
      "v_s"_a,
      "v_b"_a,
      nb::kw_only(),
      "scale"_a,
      "group_size"_a = 64,
      "bits"_a = 4,
      "mask"_a = nb::none(),
      "Fused O = softmax(Q Kᵀ·scale) V with K,V in MLX int4/int8 group format.");

  // Cheap, stream-free capability probe the Python wrapper calls before dispatching.
  m.def(
      "supports",
      [](nb::object /*q_dtype*/, int head_dim, int group_size, int bits, bool /*has_mask*/) {
        const bool d_ok = head_dim == 128;
        const bool b_ok = bits == 8 || bits == 4;  // 8-bit default target; 4-bit optional
        const bool g_ok = group_size == 32 || group_size == 64 || group_size == 128;
        return d_ok && b_ok && g_ok;
      },
      "q_dtype"_a,
      "head_dim"_a,
      "group_size"_a,
      "bits"_a,
      "has_mask"_a);
}
