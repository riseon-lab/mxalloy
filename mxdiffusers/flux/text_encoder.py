"""Qwen3 text encoder for FLUX.2-klein (ported from mflux; see PROVENANCE.md).

A faithful port of the Qwen3 decoder-stack text encoder: token embedding, N decoder
layers (RMSNorm -> grouped-query attention with rotary embeddings -> RMSNorm -> SwiGLU
MLP, each residual), a final RMSNorm, and prompt-embedding extraction that stacks
selected hidden-state layers into the transformer's context dimension.

We only ever encode a full sequence in one pass, so the KV-cache machinery from the
reference is omitted. Attribute names mirror the reference exactly so a klein checkpoint's
weight keys map without translation. INTERNAL: not part of the public API; requires mlx.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array, unsqueeze_dim: int = 1
):
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = mx.expand_dims(hidden_states, axis=2)
    hidden_states = mx.broadcast_to(hidden_states, (batch, num_kv_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.eps)
        return (self.weight.astype(mx.float32) * hidden_states).astype(input_dtype)


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_bias: bool = False,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = 1.0 / math.sqrt(head_dim)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        self.q_norm = Qwen3RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=rms_norm_eps)

    def __call__(self, hidden_states: mx.array, attention_mask, position_embeddings) -> mx.array:
        bsz, q_len, _ = hidden_states.shape
        query = self.q_proj(hidden_states).reshape(
            bsz, q_len, self.num_attention_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        query = self.q_norm(query)
        key = self.k_norm(key)

        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query, key = _apply_rotary_pos_emb(query, key, cos, sin)

        if self.num_key_value_heads != self.num_attention_heads:
            key = _repeat_kv(key, self.num_key_value_groups)
            value = _repeat_kv(value, self.num_key_value_groups)

        attn_mask = attention_mask
        if attn_mask is not None:
            attn_mask = attn_mask[:, :, :, : key.shape[2]]

        attn_output = scaled_dot_product_attention(
            query.astype(mx.float32),
            key.astype(mx.float32),
            value.astype(mx.float32),
            scale=self.scaling,
            mask=attn_mask,
        )
        attn_output = attn_output.astype(query.dtype).transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(bsz, q_len, self.num_attention_heads * self.head_dim)
        return self.o_proj(attn_output)


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_bias: bool,
        rms_norm_eps: float,
        intermediate_size: int,
    ):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            attention_bias=attention_bias,
            rms_norm_eps=rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen3MLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def __call__(self, hidden_states: mx.array, attention_mask, position_embeddings) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(
            hidden_states, attention_mask, position_embeddings
        )
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Qwen3RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 1000000.0, scaling_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.scaling_factor = scaling_factor
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))

    def __call__(self, x: mx.array, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        if position_ids.ndim == 1:
            position_ids = mx.expand_dims(position_ids, axis=0)
        inv_freq = mx.expand_dims(mx.expand_dims(self.inv_freq, axis=0), axis=0)
        pos = mx.expand_dims(position_ids.astype(mx.float32), axis=-1)
        freqs = pos * inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb) * self.scaling_factor
        sin = mx.sin(emb) * self.scaling_factor
        return cos.astype(x.dtype), sin.astype(x.dtype)


class Qwen3TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 9728,
        max_position_embeddings: int = 40960,
        rope_theta: float = 1000000.0,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 128,
        attention_bias: bool = False,
        mrope_section: list[int] | None = None,
        attention_scaling: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                attention_bias=attention_bias,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(
            dim=head_dim, base=rope_theta, scaling_factor=attention_scaling
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[mx.array, list[mx.array] | None]:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)

        mask_dtype = hidden_states.dtype
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros(attention_mask.shape, dtype=mask_dtype),
            mx.full(attention_mask.shape, -float("inf"), dtype=mask_dtype),
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)

        if seq_len == 1:
            causal_tri_mask = mx.zeros((batch_size, 1, 1, 1), dtype=mask_dtype)
        else:
            idx = mx.arange(seq_len, dtype=mx.int32)
            tri_bool = mx.expand_dims(idx, axis=0) > mx.expand_dims(idx, axis=1)
            causal_2d = mx.where(
                tri_bool,
                mx.full((seq_len, seq_len), -float("inf"), dtype=mask_dtype),
                mx.zeros((seq_len, seq_len), dtype=mask_dtype),
            )
            causal_tri_mask = mx.expand_dims(mx.expand_dims(causal_2d, axis=0), axis=0)
            causal_tri_mask = mx.broadcast_to(causal_tri_mask, (batch_size, 1, seq_len, seq_len))
        attention_mask_4d = causal_tri_mask + padding_mask

        position_ids = mx.broadcast_to(
            mx.expand_dims(mx.arange(seq_len, dtype=mx.int32), axis=0), (batch_size, seq_len)
        )
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        hidden_states_list = [hidden_states] if output_hidden_states else None
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask_4d, position_embeddings)
            if output_hidden_states:
                hidden_states_list.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        return hidden_states, hidden_states_list

    def get_prompt_embeds(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        hidden_state_layers: tuple[int, ...] = (9, 18, 27),
    ) -> mx.array:
        _, hidden_states_list = self(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        if hidden_states_list is None:
            raise RuntimeError("Hidden states not available for prompt embedding.")
        stacked = mx.stack([hidden_states_list[i] for i in hidden_state_layers], axis=1)
        batch_size, num_layers, seq_len, hidden_dim = stacked.shape
        return mx.transpose(stacked, (0, 2, 1, 3)).reshape(
            batch_size, seq_len, num_layers * hidden_dim
        )
