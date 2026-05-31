// Fused quantized-KV scaled-dot-product attention -- MLX custom primitive (impl).
//
// SKELETON. The op/encoder calls below follow MLX 0.31's metal backend, but signatures do
// drift -- verify against the installed headers (<site-packages>/mlx/include/mlx/...) and
// the canonical `mlx/examples/extensions` ("axpby") before building. This machine has no
// Metal toolchain, so none of this has been compiled here.

#include "quantized_sdpa.h"

#include <dlfcn.h>

#include <sstream>
#include <string>
#include <typeinfo>

#include <mlx/backend/metal/device.h>
#include <mlx/backend/metal/utils.h>  // type_to_name
#include <mlx/fast.h>                  // fast::scaled_dot_product_attention
#include <mlx/ops.h>                   // dequantize
#include <mlx/utils.h>                 // to_stream

namespace mxalloy::ext {

namespace {
// Anchor whose address resolves to this .so, so we can locate the metallib that the build
// places beside it (get_kernel needs the library loaded from an explicit path).
void mxalloy_anchor() {}

std::string this_lib_dir() {
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&mxalloy_anchor), &info) && info.dli_fname) {
    std::string p(info.dli_fname);
    auto pos = p.find_last_of('/');
    return pos == std::string::npos ? std::string(".") : p.substr(0, pos);
  }
  return ".";
}
}  // namespace

// ---- public factory: assemble inputs + fallback graph, return the node -----------------
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
    const std::optional<array>& mask,
    StreamOrDevice s_) {
  auto s = to_stream(s_);

  // Pure-MLX fallback: dequantize K/V then the built-in fused SDPA. Correct everywhere and
  // the numerical oracle; it pays the transient dequant spike the kernel is built to avoid.
  auto fallback =
      [scale, group_size, bits, has_mask = mask.has_value()](
          std::vector<array> in) -> std::vector<array> {
    auto k = dequantize(in[1], in[2], in[3], group_size, bits);
    auto v = dequantize(in[4], in[5], in[6], group_size, bits);
    std::optional<array> m = has_mask ? std::optional<array>(in[7]) : std::nullopt;
    // signature: (q, k, v, scale, mask_mode, mask_arr, sinks, stream)
    return {fast::scaled_dot_product_attention(in[0], k, v, scale, "", m)};
  };

  std::vector<array> inputs = {q, k_w, k_s, k_b, v_w, v_s, v_b};
  if (mask) {
    inputs.push_back(*mask);
  }

  const int head_dim = static_cast<int>(q.shape().back());
  if (QuantizedScaledDotProductAttention::use_fallback(
          q, head_dim, group_size, bits, mask.has_value(), s)) {
    return fallback(inputs)[0];
  }

  return array(
      q.shape(),
      q.dtype(),
      std::make_shared<QuantizedScaledDotProductAttention>(
          s, scale, group_size, bits, mask.has_value()),
      inputs);
}

bool QuantizedScaledDotProductAttention::use_fallback(
    const array& q, int head_dim, int group_size, int bits, bool has_mask, Stream s) {
  if (s.device == Device::cpu || has_mask) {
    return true;  // v1: no CPU kernel, no masking yet
  }
  // Only the configs with a matching [[host_name]] in quantized_sdpa.metal can run.
  const bool dt = (q.dtype() == float16);  // v1 fp16 only; bfloat + the MMA pass come next
  const bool d_ok = (head_dim == 128);
  const bool b_ok = (bits == 8 || bits == 4);  // 8-bit is the default target; 4-bit optional
  const bool g_ok = (group_size == 32 || group_size == 64 || group_size == 128);
  return !(dt && d_ok && b_ok && g_ok);
}

bool QuantizedScaledDotProductAttention::is_equivalent(const Primitive& other) const {
  if (typeid(*this) != typeid(other)) {
    return false;
  }
  const auto& o = static_cast<const QuantizedScaledDotProductAttention&>(other);
  return scale_ == o.scale_ && group_size_ == o.group_size_ && bits_ == o.bits_ &&
      has_mask_ == o.has_mask_;
}

// ---- GPU: dispatch the fused metallib kernel -------------------------------------------
void QuantizedScaledDotProductAttention::eval_gpu(
    const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& out = outputs[0];
  out.set_data(allocator::malloc(out.nbytes()));

  const auto& q = inputs[0];
  const auto qs = q.shape();
  const int B = static_cast<int>(qs[0]);
  const int H = static_cast<int>(qs[qs.size() - 3]);
  const int L = static_cast<int>(qs[qs.size() - 2]);
  const int D = static_cast<int>(qs[qs.size() - 1]);
  const int S = static_cast<int>(inputs[2].shape()[inputs[2].ndim() - 2]);  // k_scales rows

  // Entry-point name must equal a [[host_name(...)]] in the .metal; load our metallib from
  // beside this .so (get_kernel's 2nd string arg is a hash name, not a library).
  std::ostringstream kn;
  kn << "mxalloy_qsdpa_" << type_to_name(q) << "_b" << bits_ << "_d" << D;
  static const std::string metallib_path = this_lib_dir() + "/mxalloy_ext.metallib";
  auto* lib = d.get_library("mxalloy_ext", metallib_path);
  auto* kernel = d.get_kernel(kn.str(), lib);

  auto& enc = metal::get_command_encoder(s);
  enc.set_compute_pipeline_state(kernel);
  for (size_t i = 0; i < inputs.size(); ++i) {
    enc.set_input_array(inputs[i], static_cast<int>(i));
  }
  enc.set_output_array(out, 7);  // buffer(7) in the kernel signature

  // Layout must match QSDPAParams in quantized_sdpa.metal.
  struct QSDPAParams {
    int B, H, L, S;
    int group_size;
    float scale;
  } params{B, H, L, S, group_size_, scale_};
  enc.set_bytes(params, 8);

  // v2 MMA: NSG=4 simdgroups (128 threads) per threadgroup; each TG covers NSG*8=32 queries.
  constexpr int kNSG = 4, kBQ = 8;
  const int q_blocks = (L + kNSG * kBQ - 1) / (kNSG * kBQ);
  MTL::Size tg = MTL::Size(kNSG * 32, 1, 1);
  MTL::Size grid = MTL::Size(q_blocks, 1, B * H);
  enc.dispatch_threadgroups(grid, tg);
}

}  // namespace mxalloy::ext
