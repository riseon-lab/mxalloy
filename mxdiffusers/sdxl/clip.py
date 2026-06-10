"""CLIP text encoders for SDXL (CLIP-L and OpenCLIP bigG), native MLX.

Independent MLX reimplementation derived from the transformers ``CLIPTextModel`` /
``CLIPTextModelWithProjection`` reference (Apache-2.0). Attribute names mirror the checkpoint
state_dict (``text_model.encoder.layers.N...``) so the weight remap is identity.

SDXL consumes the *penultimate* layer's hidden states (no final_layer_norm) from both
encoders, and bigG's projected pooled output. Pooling uses the argmax-EOT path: these
checkpoints carry the historical ``eos_token_id=2`` config, for which transformers selects
the highest-id token — CLIP's real EOT (49407) is the vocab max, so argmax finds it.

INTERNAL: requires mlx.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass(frozen=True, slots=True)
class CLIPTextConfig:
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    hidden_act: str  # "quick_gelu" (CLIP-L) | "gelu" (bigG)
    projection_dim: int | None = None  # set -> CLIPTextModelWithProjection
    vocab_size: int = 49408
    max_position_embeddings: int = 77


CLIP_L = CLIPTextConfig(768, 12, 12, 3072, "quick_gelu")
CLIP_BIGG = CLIPTextConfig(1280, 32, 20, 5120, "gelu", projection_dim=1280)


def _act(name: str):
    if name == "quick_gelu":
        return lambda x: x * mx.sigmoid(1.702 * x)
    return nn.gelu  # transformers "gelu" is erf-gelu


class CLIPAttention(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        b, s, d = x.shape
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(b, s, d))


class CLIPMLP(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self._act = _act(cfg.hidden_act)
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self._act(self.fc1(x)))


class CLIPLayer(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(cfg.hidden_size)
        self.self_attn = CLIPAttention(cfg)
        self.layer_norm2 = nn.LayerNorm(cfg.hidden_size)
        self.mlp = CLIPMLP(cfg)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.self_attn(self.layer_norm1(x), mask)
        return x + self.mlp(self.layer_norm2(x))


class CLIPEmbeddings(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.position_embedding = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)

    def __call__(self, input_ids: mx.array) -> mx.array:
        positions = mx.arange(input_ids.shape[1])
        return self.token_embedding(input_ids) + self.position_embedding(positions)


class CLIPEncoder(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.layers = [CLIPLayer(cfg) for _ in range(cfg.num_layers)]


class CLIPTextTransformer(nn.Module):
    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.embeddings = CLIPEmbeddings(cfg)
        self.encoder = CLIPEncoder(cfg)
        self.final_layer_norm = nn.LayerNorm(cfg.hidden_size)


class CLIPTextEncoder(nn.Module):
    """One SDXL text encoder. ``__call__`` returns (penultimate_hidden, pooled | None)."""

    def __init__(self, cfg: CLIPTextConfig):
        super().__init__()
        self.cfg = cfg
        self.text_model = CLIPTextTransformer(cfg)
        if cfg.projection_dim is not None:
            self.text_projection = nn.Linear(cfg.hidden_size, cfg.projection_dim, bias=False)

    def __call__(self, input_ids: mx.array) -> tuple[mx.array, mx.array | None]:
        s = input_ids.shape[1]
        # CLIP text attention is causal.
        mask = mx.triu(mx.full((s, s), -mx.inf, dtype=mx.float32), k=1)
        x = self.text_model.embeddings(input_ids)
        mask = mask.astype(x.dtype)
        penultimate = None
        for i, layer in enumerate(self.text_model.encoder.layers):
            if i == len(self.text_model.encoder.layers) - 1:
                penultimate = x  # hidden_states[-2]: input to the last layer
            x = layer(x, mask)
        assert penultimate is not None
        if self.cfg.projection_dim is None:
            return penultimate, None
        last = self.text_model.final_layer_norm(x)
        eot = mx.argmax(input_ids, axis=-1)  # argmax-EOT pooling (see module docstring)
        pooled = mx.take_along_axis(last, eot[:, None, None], axis=1).squeeze(1)
        return penultimate, self.text_projection(pooled)
