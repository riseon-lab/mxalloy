"""FLUX.2-klein transformer (ported from mflux; see PROVENANCE.md).

A faithful port of the FLUX.2 MMDiT: a stack of double-stream blocks (joint image/text
attention) followed by single-stream blocks (parallel self-attention + MLP), with RoPE
position embeddings, timestep/guidance conditioning, and adaptive layer norm.

Attribute names mirror the FLUX.2 reference exactly so a klein checkpoint's weight keys
map without translation. INTERNAL: not part of the public API; requires mlx.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

# klein weights are bfloat16.
PRECISION = mx.bfloat16


# --- attention helpers ---------------------------------------------------------------


def _process_qkv(hidden_states, to_q, to_k, to_v, norm_q, norm_k, num_heads, head_dim):
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    query = to_q(hidden_states)
    key = to_k(hidden_states)
    value = to_v(hidden_states)
    query = mx.transpose(
        mx.reshape(query, (batch_size, seq_len, num_heads, head_dim)), (0, 2, 1, 3)
    )
    key = mx.transpose(mx.reshape(key, (batch_size, seq_len, num_heads, head_dim)), (0, 2, 1, 3))
    value = mx.transpose(
        mx.reshape(value, (batch_size, seq_len, num_heads, head_dim)), (0, 2, 1, 3)
    )
    q_dtype = query.dtype
    k_dtype = key.dtype
    query = norm_q(query.astype(mx.float32)).astype(q_dtype)
    key = norm_k(key.astype(mx.float32)).astype(k_dtype)
    return query, key, value


def _compute_attention(query, key, value, batch_size, num_heads, head_dim, mask=None):
    scale = 1 / mx.sqrt(query.shape[-1])
    hidden_states = scaled_dot_product_attention(query, key, value, scale=scale, mask=mask)
    hidden_states = mx.transpose(hidden_states, (0, 2, 1, 3))
    hidden_states = mx.reshape(hidden_states, (batch_size, -1, num_heads * head_dim))
    return hidden_states


def _apply_rope_bshd(xq, xk, cos, sin):
    out_dtype = xq.dtype
    xq_f = xq.astype(mx.float32)
    xk_f = xk.astype(mx.float32)
    cos_b = cos.reshape(1, 1, cos.shape[0], cos.shape[1])
    sin_b = sin.reshape(1, 1, sin.shape[0], sin.shape[1])

    def mix(x):
        x2 = x.reshape(*x.shape[:-1], -1, 2)
        real = x2[..., 0]
        imag = x2[..., 1]
        out0 = real * cos_b + (-imag) * sin_b
        out1 = imag * cos_b + real * sin_b
        out2 = mx.stack([out0, out1], axis=-1)
        return out2.reshape(*x.shape)

    return mix(xq_f).astype(out_dtype), mix(xk_f).astype(out_dtype)


# --- embeddings / conditioning -------------------------------------------------------


class Flux2PosEmbed(nn.Module):
    def __init__(self, theta: int = 2000, axes_dim: tuple[int, ...] = (32, 32, 32, 32)):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: mx.array) -> tuple[mx.array, mx.array]:
        cos_out = []
        sin_out = []
        pos = ids.astype(mx.float32)
        for i, dim in enumerate(self.axes_dim):
            cos, sin = self._get_1d_rope(dim, pos[..., i])
            cos_out.append(cos)
            sin_out.append(sin)
        return mx.concatenate(cos_out, axis=-1), mx.concatenate(sin_out, axis=-1)

    def _get_1d_rope(self, dim: int, pos: mx.array) -> tuple[mx.array, mx.array]:
        scale = mx.arange(0, dim, 2, dtype=mx.float32) / dim
        omega = 1.0 / (self.theta**scale)
        out = mx.expand_dims(pos, axis=-1) * mx.expand_dims(omega, axis=0)
        return mx.cos(out), mx.sin(out)


class Flux2TimestepGuidanceEmbeddings(nn.Module):
    def __init__(
        self, in_channels: int = 256, embedding_dim: int = 3072, guidance_embeds: bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.guidance_embeds = guidance_embeds
        self.linear_1 = nn.Linear(in_channels, embedding_dim, bias=False)
        self.linear_2 = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.guidance_linear_1 = (
            nn.Linear(in_channels, embedding_dim, bias=False) if guidance_embeds else None
        )
        self.guidance_linear_2 = (
            nn.Linear(embedding_dim, embedding_dim, bias=False) if guidance_embeds else None
        )

    def __call__(self, timestep: mx.array, guidance: mx.array | None) -> mx.array:
        timestep = self._timestep_embedding(timestep.astype(mx.float32), self.in_channels)
        timesteps_emb = self.linear_2(nn.silu(self.linear_1(timestep)))
        if (
            guidance is not None
            and self.guidance_linear_1 is not None
            and self.guidance_linear_2 is not None
        ):
            guidance = self._timestep_embedding(guidance.astype(mx.float32), self.in_channels)
            guidance_emb = self.guidance_linear_2(nn.silu(self.guidance_linear_1(guidance)))
            return timesteps_emb + guidance_emb
        return timesteps_emb

    @staticmethod
    def _timestep_embedding(
        timesteps: mx.array, dim: int, flip_sin_to_cos: bool = True
    ) -> mx.array:
        half = dim // 2
        freqs = mx.exp(-math.log(10000.0) * mx.arange(0, half, dtype=mx.float32) / half)
        args = timesteps[:, None] * freqs[None, :]
        emb = mx.concatenate([mx.sin(args), mx.cos(args)], axis=-1)
        if flip_sin_to_cos:
            emb = mx.concatenate([emb[:, half:], emb[:, :half]], axis=-1)
        if dim % 2 == 1:
            emb = mx.concatenate([emb, mx.zeros((emb.shape[0], 1), dtype=emb.dtype)], axis=-1)
        return emb


class Flux2Modulation(nn.Module):
    def __init__(self, dim: int, mod_param_sets: int = 2):
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.linear = nn.Linear(dim, dim * 3 * mod_param_sets, bias=False)

    def __call__(self, temb: mx.array):
        mod = self.linear(nn.silu(temb))
        if mod.ndim == 2:
            mod = mx.expand_dims(mod, axis=1)
        mod_params = mx.split(mod, 3 * self.mod_param_sets, axis=-1)
        return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(self.mod_param_sets))


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=False)
        self.norm = nn.LayerNorm(dims=embedding_dim, eps=1e-6, affine=False)

    def __call__(self, x: mx.array, text_embeddings: mx.array) -> mx.array:
        text_embeddings = self.linear(nn.silu(text_embeddings).astype(PRECISION))
        chunk = self.embedding_dim
        scale = text_embeddings[:, 0 * chunk : 1 * chunk]
        shift = text_embeddings[:, 1 * chunk : 2 * chunk]
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


# --- feed-forward --------------------------------------------------------------------


class Flux2SwiGLU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        x1, x2 = mx.split(x, 2, axis=-1)
        return nn.silu(x1) * x2


class Flux2FeedForward(nn.Module):
    def __init__(self, dim: int, mult: float = 3.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=False)
        self.act = Flux2SwiGLU()
        self.linear_out = nn.Linear(inner_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear_out(self.act(self.linear_in(x)))


# --- attention modules ---------------------------------------------------------------


class Flux2Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, added_kv_proj_dim: int | None = None):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.added_kv_proj_dim = added_kv_proj_dim
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=1e-5)
        self.norm_k = nn.RMSNorm(dim_head, eps=1e-5)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)
        if added_kv_proj_dim is not None:
            self.norm_added_q = nn.RMSNorm(dim_head, eps=1e-5)
            self.norm_added_k = nn.RMSNorm(dim_head, eps=1e-5)
            self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=False)
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=False)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim, bias=False)
            self.to_add_out = nn.Linear(self.inner_dim, dim, bias=False)

    def __call__(self, hidden_states: mx.array, encoder_hidden_states: mx.array, image_rotary_emb):
        query, key, value = _process_qkv(
            hidden_states,
            self.to_q,
            self.to_k,
            self.to_v,
            self.norm_q,
            self.norm_k,
            self.heads,
            self.dim_head,
        )
        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            enc_query, enc_key, enc_value = _process_qkv(
                encoder_hidden_states,
                self.add_q_proj,
                self.add_k_proj,
                self.add_v_proj,
                self.norm_added_q,
                self.norm_added_k,
                self.heads,
                self.dim_head,
            )
            query = mx.concatenate([enc_query, query], axis=2)
            key = mx.concatenate([enc_key, key], axis=2)
            value = mx.concatenate([enc_value, value], axis=2)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            query, key = _apply_rope_bshd(query, key, cos, sin)

        hidden_states = _compute_attention(
            query, key, value, hidden_states.shape[0], self.heads, self.dim_head
        )

        if encoder_hidden_states is not None and self.added_kv_proj_dim is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

        return self.to_out(hidden_states), encoder_hidden_states


class Flux2ParallelSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, mlp_ratio: float = 3.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.to_qkv_mlp_proj = nn.Linear(
            dim, self.inner_dim * 3 + self.mlp_hidden_dim * 2, bias=False
        )
        self.norm_q = nn.RMSNorm(dim_head, eps=1e-5)
        self.norm_k = nn.RMSNorm(dim_head, eps=1e-5)
        self.mlp_act = Flux2SwiGLU()
        self.to_out = nn.Linear(self.inner_dim + self.mlp_hidden_dim, dim, bias=False)

    def __call__(self, hidden_states: mx.array, image_rotary_emb):
        proj = self.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden = mx.split(proj, [self.inner_dim * 3], axis=-1)
        query, key, value = mx.split(qkv, 3, axis=-1)

        batch, seq_len, _ = query.shape
        query = mx.transpose(
            mx.reshape(query, (batch, seq_len, self.heads, self.dim_head)), (0, 2, 1, 3)
        )
        key = mx.transpose(
            mx.reshape(key, (batch, seq_len, self.heads, self.dim_head)), (0, 2, 1, 3)
        )
        value = mx.transpose(
            mx.reshape(value, (batch, seq_len, self.heads, self.dim_head)), (0, 2, 1, 3)
        )

        query = self.norm_q(query.astype(mx.float32)).astype(PRECISION)
        key = self.norm_k(key.astype(mx.float32)).astype(PRECISION)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            query, key = _apply_rope_bshd(query, key, cos, sin)

        hidden_states = _compute_attention(query, key, value, batch, self.heads, self.dim_head)
        mlp_hidden = self.mlp_act(mlp_hidden)
        hidden_states = mx.concatenate([hidden_states, mlp_hidden], axis=-1)
        return self.to_out(hidden_states)


# --- blocks --------------------------------------------------------------------------


class Flux2TransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 3.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.norm1_context = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.attn = Flux2Attention(
            dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, added_kv_proj_dim=dim
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.ff = Flux2FeedForward(dim=dim, mult=mlp_ratio)
        self.norm2_context = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.ff_context = Flux2FeedForward(dim=dim, mult=mlp_ratio)

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states,
        temb_mod_params_img,
        temb_mod_params_txt,
        image_rotary_emb,
    ):
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = (
            temb_mod_params_txt
        )

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

        attn_output, encoder_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_output
        encoder_hidden_states = encoder_hidden_states + c_gate_msa * encoder_attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = (1 + scale_mlp) * norm_hidden_states + shift_mlp
        hidden_states = hidden_states + gate_mlp * self.ff(norm_hidden_states)

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = (1 + c_scale_mlp) * norm_encoder_hidden_states + c_shift_mlp
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp * self.ff_context(
            norm_encoder_hidden_states
        )

        return encoder_hidden_states, hidden_states


class Flux2SingleTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, mlp_ratio: float = 3.0
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.attn = Flux2ParallelSelfAttention(
            dim=dim, heads=num_attention_heads, dim_head=attention_head_dim, mlp_ratio=mlp_ratio
        )

    def __call__(self, hidden_states, temb_mod_params, image_rotary_emb):
        mod_shift, mod_scale, mod_gate = temb_mod_params
        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift
        attn_output = self.attn(norm_hidden_states, image_rotary_emb)
        return hidden_states + mod_gate * attn_output


# --- top-level transformer -----------------------------------------------------------


class Flux2Transformer(nn.Module):
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: int | None = None,
        num_layers: int = 5,
        num_single_layers: int = 20,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 7680,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        guidance_embeds: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            guidance_embeds=guidance_embeds,
        )
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2)
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1)

        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)
        self.transformer_blocks = [
            Flux2TransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_layers)
        ]
        self.single_transformer_blocks = [
            Flux2SingleTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(num_single_layers)
        ]
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * self.out_channels, bias=False
        )
        self._cached_rope = None
        self._cached_rope_keys = None

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array | float | int,
        img_ids: mx.array,
        txt_ids: mx.array,
        guidance: mx.array | float | int | None = None,
    ) -> mx.array:
        if not isinstance(timestep, mx.array):
            timestep = mx.array(timestep, dtype=hidden_states.dtype)
        if timestep.ndim == 0:
            timestep = mx.full((hidden_states.shape[0],), timestep, dtype=hidden_states.dtype)
        timestep = timestep.astype(hidden_states.dtype)
        timestep_scale = mx.where(mx.max(timestep) <= 1.0, 1000.0, 1.0).astype(hidden_states.dtype)
        timestep = timestep * timestep_scale
        if guidance is not None:
            if not isinstance(guidance, mx.array):
                guidance = mx.array(guidance, dtype=hidden_states.dtype)
            if guidance.ndim == 0:
                guidance = mx.full((hidden_states.shape[0],), guidance, dtype=hidden_states.dtype)
            guidance = guidance.astype(hidden_states.dtype)
            guidance_scale = mx.where(mx.max(guidance) <= 1.0, 1000.0, 1.0).astype(
                hidden_states.dtype
            )
            guidance = guidance * guidance_scale
        temb = self.time_guidance_embed(timestep, guidance).astype(PRECISION)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        if (
            getattr(self, "_cached_rope", None) is not None
            and getattr(self, "_cached_rope_keys", None) is not None
            and self._cached_rope_keys[0] is img_ids
            and self._cached_rope_keys[1] is txt_ids
        ):
            concat_rotary_emb = self._cached_rope
        else:
            image_rotary_emb = self.pos_embed(img_ids)
            text_rotary_emb = self.pos_embed(txt_ids)
            concat_rotary_emb = (
                mx.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
                mx.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
            )
            mx.eval(concat_rotary_emb[0], concat_rotary_emb[1])
            self._cached_rope = concat_rotary_emb
            self._cached_rope_keys = (img_ids, txt_ids)

        temb_mod_params_img = self.double_stream_modulation_img(temb)
        temb_mod_params_txt = self.double_stream_modulation_txt(temb)
        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb_mod_params_img=temb_mod_params_img,
                temb_mod_params_txt=temb_mod_params_txt,
                image_rotary_emb=concat_rotary_emb,
            )

        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
        temb_mod_params_single = self.single_stream_modulation(temb)[0]
        for block in self.single_transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                temb_mod_params=temb_mod_params_single,
                image_rotary_emb=concat_rotary_emb,
            )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)
