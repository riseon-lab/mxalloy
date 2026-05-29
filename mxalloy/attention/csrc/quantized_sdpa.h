// Fused quantized-KV scaled-dot-product attention -- MLX custom primitive (declaration).
//
// SKELETON. Structure is grounded in the installed MLX 0.31 headers
// (mlx/fast_primitives.h: `ScaledDotProductAttention`/`RoPE` derive from `fast::Custom`),
// but exact backend-encoder signatures evolve between MLX releases -- before building,
// diff this against the canonical extension example that ships with MLX
// (`mlx/examples/extensions`, the "axpby" primitive) and the headers under
// `<site-packages>/mlx/include/mlx/`.
//
// Why `fast::Custom`: it carries a *fallback* graph (here: dequantize K/V -> SDPA) that MLX
// runs whenever `eval_gpu` is inapplicable (CPU stream, unsupported dtype/head_dim/bits).
// So the op is always correct; the metallib kernel is a spike-free fast path layered on top.

#pragma once

#include <functional>
#include <optional>
#include <vector>

#include <mlx/fast_primitives.h>

namespace mxalloy::ext {

using namespace mlx::core;

// Public op -- builds the graph node. K and V are passed as the three arrays that
// `mx.quantize` produces (packed uint32 weights, per-group scales, per-group biases),
// quantized along head_dim. Inputs order is fixed: q, kw, ks, kb, vw, vs, vb, [mask].
array quantized_scaled_dot_product_attention(
    const array& q,
    const array& k_w,
    const array& k_s,
    const array& k_b,
    const array& v_w,
    const array& v_s,
    const array& v_b,
    float scale,
    int group_size,
    int bits,
    const std::optional<array>& mask = std::nullopt,
    StreamOrDevice s = {});

class QuantizedScaledDotProductAttention : public fast::Custom {
 public:
  QuantizedScaledDotProductAttention(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float scale,
      int group_size,
      int bits,
      bool has_mask)
      : Custom(stream, std::move(fallback)),
        scale_(scale),
        group_size_(group_size),
        bits_(bits),
        has_mask_(has_mask) {}

  // CPU is served by the fallback graph; only the GPU fast path is custom.
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error(
        "QuantizedSDPA: no CPU kernel; the fallback graph handles CPU.");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  bool is_equivalent(const Primitive& other) const override;

  // Output is shaped like q (inputs[0]); MLX provides this macro for exactly that case.
  DEFINE_INPUT_OUTPUT_SHAPE()
  DEFINE_NAME(QuantizedScaledDotProductAttention)

  // Decides at trace time whether the kernel can run, else MLX takes the fallback.
  static bool use_fallback(
      const array& q,
      int head_dim,
      int group_size,
      int bits,
      bool has_mask,
      Stream s);

 private:
  float scale_;
  int group_size_;
  int bits_;
  bool has_mask_;
};

}  // namespace mxalloy::ext
