"""Qwen3 text encoder for the FLUX.2 family (and Z-Image), native MLX.

Independent MLX reimplementation derived from the transformers ``Qwen3ForCausalLM``
reference (Apache-2.0, attributed in ``NOTICE``). Attribute names mirror the checkpoint
state_dict (``model.layers.N...``) so the weight remap is identity. klein-4B config:
36 layers, hidden 2560, GQA 32 query / 8 kv heads x 128 head_dim with per-head q/k RMSNorm,
SwiGLU mlp 9728, RMSNorm eps 1e-6, rope theta 1e6, tied embeddings (no lm_head; this class
is encoder-only and never computes logits).

``__call__`` mirrors transformers' hidden-states convention: returns
``(final_normed_hidden, hidden_states)`` where ``hidden_states[0]`` is the embedding output,
``hidden_states[k]`` is layer k's output, and the final entry has ``model.norm`` applied —
so diffusers-style ``hidden_states[9/18/27]`` and ``hidden_states[-2]`` index identically.

INTERNAL: requires mlx.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True, slots=True)
class Qwen3Config:
    hidden_size: int = 2560
    num_layers: int = 36
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 9728
    vocab_size: int = 151936
    rms_eps: float = 1e-6
    rope_theta: float = 1_000_000.0


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_eps)
        self.k_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_eps)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        cfg = self.cfg
        b, s, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, s, cfg.num_heads, cfg.head_dim))
        k = self.k_norm(self.k_proj(x).reshape(b, s, cfg.num_kv_heads, cfg.head_dim))
        v = self.v_proj(x).reshape(b, s, cfg.num_kv_heads, cfg.head_dim)
        q = mx.fast.rope(
            q.transpose(0, 2, 1, 3), cfg.head_dim, traditional=False, base=cfg.rope_theta,
            scale=1.0, offset=0,
        )
        k = mx.fast.rope(
            k.transpose(0, 2, 1, 3), cfg.head_dim, traditional=False, base=cfg.rope_theta,
            scale=1.0, offset=0,
        )
        v = v.transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=cfg.head_dim**-0.5, mask=mask)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(b, s, -1))


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Layer(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)
        self.self_attn = Qwen3Attention(cfg)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)
        self.mlp = Qwen3MLP(cfg)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.self_attn(self.input_layernorm(x), mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class _Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = [Qwen3Layer(cfg) for _ in range(cfg.num_layers)]
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_eps)


class Qwen3TextEncoder(nn.Module):
    def __init__(self, cfg: Qwen3Config | None = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else Qwen3Config()
        self.model = _Qwen3Model(self.cfg)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        output_hidden_states: bool = True,
    ) -> tuple[mx.array, list[mx.array]]:
        s = input_ids.shape[1]
        causal = mx.triu(mx.full((s, s), -mx.inf, dtype=mx.float32), k=1)
        if attention_mask is not None:
            pad = mx.where(attention_mask[:, None, None, :] == 0, -mx.inf, 0.0)
            mask = causal[None, None] + pad
        else:
            mask = causal[None, None]
        x = self.model.embed_tokens(input_ids)
        mask = mask.astype(x.dtype)
        hidden: list[mx.array] = [x]
        for layer in self.model.layers:
            x = layer(x, mask)
            hidden.append(x)
        x = self.model.norm(x)
        hidden[-1] = x  # transformers convention: the final entry is post-norm
        return x, hidden

    def get_prompt_embeds(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None,
        out_layers: tuple[int, ...],
    ) -> mx.array:
        """Concat ``hidden_states[k]`` for k in out_layers featurewise -> (B, S, len*hidden)."""
        _, hidden = self(input_ids, attention_mask)
        return mx.concatenate([hidden[k] for k in out_layers], axis=-1)
