"""FLUX.2 MMDiT transformer (klein), native MLX.

Independent MLX reimplementation derived from the diffusers ``Flux2Transformer2DModel``
reference (Apache-2.0, attributed in ``NOTICE``) and the FLUX.2-klein-4B checkpoint.
Attribute names mirror the checkpoint state_dict, so the weight remap is identity.

klein-4B config facts: 5 double-stream + 20 single-stream blocks; 24 heads x 128 dim (3072);
SwiGLU mlp_ratio 3 fused into the in-projections; modulation factored *globally* — one
silu+linear AdaLN per stream (img/txt/single) computed once per step from the timestep
embedding and shared by every block of that stream; per-head RMSNorm on q/k; 4-axis RoPE
(T, H, W, L), 32 dims each, theta 2000; text tokens lead the joint sequence; bias-free
linears throughout; ``guidance_embeds`` false (the distilled klein ignores guidance).

INTERNAL: requires mlx.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

_DIM = 3072
_HEADS = 24
_HEAD_DIM = 128
_NUM_DOUBLE = 5
_NUM_SINGLE = 20
_MLP_RATIO = 3.0
_JOINT_DIM = 7680  # 3 stacked Qwen3 hidden layers (3 x 2560)
_IN_CHANNELS = 128  # 32-ch VAE latents, 2x2 patched
_TIME_CHANNELS = 256
_ROPE_AXES = (32, 32, 32, 32)
_ROPE_THETA = 2000
_EPS = 1e-6


def timestep_sinusoid(t: mx.array, dim: int = _TIME_CHANNELS) -> mx.array:
    """Sinusoidal embedding, diffusers Timesteps convention (flip_sin_to_cos, shift 0)."""
    half = dim // 2
    exponent = -math.log(10000) * mx.arange(half, dtype=mx.float32) / half
    emb = t.astype(mx.float32)[:, None] * mx.exp(exponent)[None, :]
    return mx.concatenate([mx.cos(emb), mx.sin(emb)], axis=-1)


def rope_frequencies(ids: mx.array) -> tuple[mx.array, mx.array]:
    """4-axis rotary cos/sin for position ids (S, 4) -> ((S, 64), (S, 64)) half-width tables.

    Tables are computed in float32 (reference precision) per pair lane; the interleaved
    duplication ([c0,c0,c1,c1,...]) is folded into :func:`_apply_rope`'s pair math instead of
    materialising full-width tables.
    """
    cos_parts, sin_parts = [], []
    pos = ids.astype(mx.float32)
    for axis, dim in enumerate(_ROPE_AXES):
        idx = mx.arange(0, dim, 2, dtype=mx.float32) / dim
        freqs = pos[:, axis : axis + 1] * (_ROPE_THETA**-idx)[None, :]  # (S, dim/2)
        cos_parts.append(mx.cos(freqs))
        sin_parts.append(mx.sin(freqs))
    return mx.concatenate(cos_parts, axis=-1), mx.concatenate(sin_parts, axis=-1)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Interleaved-pair rotation on (B, H, S, D); cos/sin (S, D/2) applied per pair lane."""
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    c = cos.astype(x.dtype)
    s = sin.astype(x.dtype)
    return mx.stack([a * c - b * s, b * c + a * s], axis=-1).reshape(x.shape)


def _split_mod(mod: mx.array, sets: int) -> list[tuple[mx.array, mx.array, mx.array]]:
    """(B, sets*3*dim) -> per set (shift, scale, gate), each (B, 1, dim)."""
    chunks = mx.split(mod[:, None, :], 3 * sets, axis=-1)
    return [tuple(chunks[3 * i : 3 * (i + 1)]) for i in range(sets)]


class TimestepEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(_TIME_CHANNELS, _DIM, bias=False)
        self.linear_2 = nn.Linear(_DIM, _DIM, bias=False)

    def __call__(self, proj: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(proj)))


class TimeGuidanceEmbed(nn.Module):
    """klein: guidance_embeds=False — the guidance arg is accepted and ignored."""

    def __init__(self):
        super().__init__()
        self.timestep_embedder = TimestepEmbedder()

    def __call__(self, timestep: mx.array, dtype: mx.Dtype) -> mx.array:
        return self.timestep_embedder(timestep_sinusoid(timestep).astype(dtype))


class Modulation(nn.Module):
    """silu(temb) -> linear; one global instance per stream, shared by all its blocks."""

    def __init__(self, sets: int):
        super().__init__()
        self.linear = nn.Linear(_DIM, _DIM * 3 * sets, bias=False)

    def __call__(self, temb: mx.array) -> mx.array:
        return self.linear(nn.silu(temb))


class SwiGLUFeedForward(nn.Module):
    """linear_in projects to 2x inner; silu(gate) * x; linear_out back to dim."""

    def __init__(self):
        super().__init__()
        inner = int(_DIM * _MLP_RATIO)
        self.linear_in = nn.Linear(_DIM, inner * 2, bias=False)
        self.linear_out = nn.Linear(inner, _DIM, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate, value = mx.split(self.linear_in(x), 2, axis=-1)
        return self.linear_out(nn.silu(gate) * value)


def _heads(x: mx.array) -> mx.array:
    b, s, _ = x.shape
    return x.reshape(b, s, _HEADS, _HEAD_DIM)


class JointAttention(nn.Module):
    """Double-stream attention: separate img/txt projections, joint SDPA, text first."""

    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(_DIM, _DIM, bias=False)
        self.to_k = nn.Linear(_DIM, _DIM, bias=False)
        self.to_v = nn.Linear(_DIM, _DIM, bias=False)
        self.add_q_proj = nn.Linear(_DIM, _DIM, bias=False)
        self.add_k_proj = nn.Linear(_DIM, _DIM, bias=False)
        self.add_v_proj = nn.Linear(_DIM, _DIM, bias=False)
        self.norm_q = nn.RMSNorm(_HEAD_DIM, eps=_EPS)
        self.norm_k = nn.RMSNorm(_HEAD_DIM, eps=_EPS)
        self.norm_added_q = nn.RMSNorm(_HEAD_DIM, eps=_EPS)
        self.norm_added_k = nn.RMSNorm(_HEAD_DIM, eps=_EPS)
        self.to_out = [nn.Linear(_DIM, _DIM, bias=False)]
        self.to_add_out = nn.Linear(_DIM, _DIM, bias=False)

    def __call__(
        self, img: mx.array, txt: mx.array, cos: mx.array, sin: mx.array
    ) -> tuple[mx.array, mx.array]:
        txt_len = txt.shape[1]
        q = mx.concatenate([self.norm_added_q(_heads(self.add_q_proj(txt))),
                            self.norm_q(_heads(self.to_q(img)))], axis=1)
        k = mx.concatenate([self.norm_added_k(_heads(self.add_k_proj(txt))),
                            self.norm_k(_heads(self.to_k(img)))], axis=1)
        v = mx.concatenate([_heads(self.add_v_proj(txt)), _heads(self.to_v(img))], axis=1)
        q = _apply_rope(q.transpose(0, 2, 1, 3), cos, sin)
        k = _apply_rope(k.transpose(0, 2, 1, 3), cos, sin)
        o = mx.fast.scaled_dot_product_attention(
            q, k, v.transpose(0, 2, 1, 3), scale=_HEAD_DIM**-0.5
        )
        o = o.transpose(0, 2, 1, 3).reshape(img.shape[0], -1, _DIM)
        return self.to_out[0](o[:, txt_len:]), self.to_add_out(o[:, :txt_len])


class TransformerBlock(nn.Module):
    """Double-stream block; modulation arrives precomputed (global, shared)."""

    def __init__(self):
        super().__init__()
        self.attn = JointAttention()
        self.ff = SwiGLUFeedForward()
        self.ff_context = SwiGLUFeedForward()
        self._norm = nn.LayerNorm(_DIM, eps=_EPS, affine=False)

    def __call__(
        self,
        img: mx.array,
        txt: mx.array,
        mod_img: mx.array,
        mod_txt: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> tuple[mx.array, mx.array]:
        (i_shift1, i_scale1, i_gate1), (i_shift2, i_scale2, i_gate2) = _split_mod(mod_img, 2)
        (t_shift1, t_scale1, t_gate1), (t_shift2, t_scale2, t_gate2) = _split_mod(mod_txt, 2)

        norm_img = (1 + i_scale1) * self._norm(img) + i_shift1
        norm_txt = (1 + t_scale1) * self._norm(txt) + t_shift1
        attn_img, attn_txt = self.attn(norm_img, norm_txt, cos, sin)

        img = img + i_gate1 * attn_img
        img = img + i_gate2 * self.ff((1 + i_scale2) * self._norm(img) + i_shift2)
        txt = txt + t_gate1 * attn_txt
        txt = txt + t_gate2 * self.ff_context((1 + t_scale2) * self._norm(txt) + t_shift2)
        return img, txt


class ParallelSelfAttention(nn.Module):
    """Single-stream fused block: one in-projection for QKV+MLP, one out for attn|mlp."""

    def __init__(self):
        super().__init__()
        inner = int(_DIM * _MLP_RATIO)
        self.to_qkv_mlp_proj = nn.Linear(_DIM, 3 * _DIM + 2 * inner, bias=False)
        self.to_out = nn.Linear(_DIM + inner, _DIM, bias=False)
        self.norm_q = nn.RMSNorm(_HEAD_DIM, eps=_EPS)
        self.norm_k = nn.RMSNorm(_HEAD_DIM, eps=_EPS)
        self._inner = inner

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        b, s, _ = x.shape
        proj = self.to_qkv_mlp_proj(x)
        qkv = proj[..., : 3 * _DIM]
        gate, value = mx.split(proj[..., 3 * _DIM :], 2, axis=-1)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = _apply_rope(self.norm_q(_heads(q)).transpose(0, 2, 1, 3), cos, sin)
        k = _apply_rope(self.norm_k(_heads(k)).transpose(0, 2, 1, 3), cos, sin)
        o = mx.fast.scaled_dot_product_attention(
            q, k, _heads(v).transpose(0, 2, 1, 3), scale=_HEAD_DIM**-0.5
        )
        o = o.transpose(0, 2, 1, 3).reshape(b, s, _DIM)
        return self.to_out(mx.concatenate([o, nn.silu(gate) * value], axis=-1))


class SingleTransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = ParallelSelfAttention()
        self._norm = nn.LayerNorm(_DIM, eps=_EPS, affine=False)

    def __call__(self, x: mx.array, mod: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        ((shift, scale, gate),) = _split_mod(mod, 1)
        return x + gate * self.attn((1 + scale) * self._norm(x) + shift, cos, sin)


class AdaLayerNormOut(nn.Module):
    """norm_out: silu(temb) -> linear -> (scale, shift); x = norm(x)*(1+scale)+shift."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(_DIM, 2 * _DIM, bias=False)
        self._norm = nn.LayerNorm(_DIM, eps=_EPS, affine=False)

    def __call__(self, x: mx.array, temb: mx.array) -> mx.array:
        scale, shift = mx.split(self.linear(nn.silu(temb))[:, None, :], 2, axis=-1)
        return (1 + scale) * self._norm(x) + shift


class Flux2Transformer(nn.Module):
    """eps = transformer(packed_latents, t, text_embeds, txt_ids, img_ids)."""

    def __init__(self) -> None:
        super().__init__()
        self.x_embedder = nn.Linear(_IN_CHANNELS, _DIM, bias=False)
        self.context_embedder = nn.Linear(_JOINT_DIM, _DIM, bias=False)
        self.time_guidance_embed = TimeGuidanceEmbed()
        self.double_stream_modulation_img = Modulation(sets=2)
        self.double_stream_modulation_txt = Modulation(sets=2)
        self.single_stream_modulation = Modulation(sets=1)
        self.transformer_blocks = [TransformerBlock() for _ in range(_NUM_DOUBLE)]
        self.single_transformer_blocks = [SingleTransformerBlock() for _ in range(_NUM_SINGLE)]
        self.norm_out = AdaLayerNormOut()
        self.proj_out = nn.Linear(_DIM, _IN_CHANNELS, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,  # (B, S_img, 128) packed latents
        timestep: mx.array,  # (B,) in [0, 1] — multiplied by 1000 here (reference convention)
        encoder_hidden_states: mx.array,  # (B, S_txt, 7680)
        txt_ids: mx.array,  # (S_txt, 4)
        img_ids: mx.array,  # (S_img, 4)
        rope: tuple[mx.array, mx.array] | None = None,  # precomputed (cos, sin), optional
    ) -> mx.array:
        dtype = hidden_states.dtype
        temb = self.time_guidance_embed(timestep * 1000.0, dtype)
        mod_img = self.double_stream_modulation_img(temb)
        mod_txt = self.double_stream_modulation_txt(temb)
        mod_single = self.single_stream_modulation(temb)

        img = self.x_embedder(hidden_states)
        txt = self.context_embedder(encoder_hidden_states)

        if rope is None:
            rope = self.compute_rope(txt_ids, img_ids)
        cos, sin = rope

        for block in self.transformer_blocks:
            img, txt = block(img, txt, mod_img, mod_txt, cos, sin)

        x = mx.concatenate([txt, img], axis=1)
        for block in self.single_transformer_blocks:
            x = block(x, mod_single, cos, sin)

        x = x[:, txt.shape[1] :]
        return self.proj_out(self.norm_out(x, temb))

    @staticmethod
    def compute_rope(txt_ids: mx.array, img_ids: mx.array) -> tuple[mx.array, mx.array]:
        """Joint rotary tables, text first — cacheable per (prompt length, image size)."""
        txt_cos, txt_sin = rope_frequencies(txt_ids)
        img_cos, img_sin = rope_frequencies(img_ids)
        return (
            mx.concatenate([txt_cos, img_cos], axis=0),
            mx.concatenate([txt_sin, img_sin], axis=0),
        )
