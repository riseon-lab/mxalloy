// Fused quantized-KV scaled-dot-product attention -- MLX custom primitive (declaration).
//
// Subclasses the exported mlx::core::Primitive directly (NOT fast::Custom -- that base is
// not MLX_API-exported, so a third-party subclass can't link). The fallback (dequantize
// K/V -> SDPA) is gated in the factory: when use_fallback() is true we return the fallback
// graph and never construct this primitive, so eval_gpu only runs for supported GPU configs.

#pragma once

#include <optional>
#include <vector>

#include <mlx/primitives.h>
#include <mlx/utils.h>  // StreamOrDevice

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

class QuantizedScaledDotProductAttention : public Primitive {
 public:
  QuantizedScaledDotProductAttention(
      Stream stream, float scale, int group_size, int bits, bool has_mask)
      : Primitive(stream),
        scale_(scale),
        group_size_(group_size),
        bits_(bits),
        has_mask_(has_mask) {}

  // Never constructed for a CPU stream (the factory falls back first), so CPU just throws.
  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::runtime_error("QuantizedSDPA: GPU-only; CPU goes through the fallback graph.");
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
