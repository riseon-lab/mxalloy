"""Z-Image single-stream DiT (NextDiT / Lumina-2 family), native MLX.

Independent MLX reimplementation derived from diffusers' ``ZImageTransformer2DModel``
reference (Apache-2.0, attributed in ``NOTICE``) for the batch-1 text-to-image path. See
``SPEC.md`` for the source-grounded architecture. The
``SEQ_MULTI_OF=32`` padding the reference uses is a no-op for a single sample under masked
attention (pad tokens are masked out and their outputs discarded), so we run the exact,
unpadded sequence ``[image_tokens, caption_tokens]`` with full attention — only the image
block's RoPE position offset (``ceil(cap_len/32)*32 + 1``) is preserved.

INTERNAL; requires mlx.
"""

from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn

SEQ_MULTI_OF = 32
ADALN_EMBED_DIM = 256


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _swiglu_hidden(dim: int) -> int:
    return int(dim / 3 * 8)


def build_rope_cos_sin(
    pos_ids: mx.array, axes_dims: list[int], theta: float
) -> tuple[mx.array, mx.array]:
    """pos_ids (L, n_axes) int -> (cos, sin) each (L, head_dim/2), per-axis concatenated."""
    cos_parts, sin_parts = [], []
    for axis, d in enumerate(axes_dims):
        inv_freq = 1.0 / (theta ** (mx.arange(0, d, 2, dtype=mx.float32) / d))  # (d/2,)
        ang = pos_ids[:, axis : axis + 1].astype(mx.float32) * inv_freq[None, :]  # (L, d/2)
        cos_parts.append(mx.cos(ang))
        sin_parts.append(mx.sin(ang))
    return mx.concatenate(cos_parts, axis=-1), mx.concatenate(sin_parts, axis=-1)  # (L, sum half)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """x (B, L, H, D); cos/sin (1, L, 1, D/2). Complex rotate consecutive pairs."""
    b, ln, h, d = x.shape
    xp = x.reshape(b, ln, h, d // 2, 2)
    xr, xi = xp[..., 0], xp[..., 1]
    out_r = xr * cos - xi * sin
    out_i = xr * sin + xi * cos
    return mx.stack([out_r, out_i], axis=-1).reshape(b, ln, h, d)


class ZImageAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, qk_norm: bool, eps: float):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = [nn.Linear(dim, dim, bias=False)]  # list -> key `to_out.0` (diffusers)
        self.norm_q = nn.RMSNorm(self.head_dim, eps=1e-5) if qk_norm else None
        self.norm_k = nn.RMSNorm(self.head_dim, eps=1e-5) if qk_norm else None

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        b, ln, _ = x.shape
        q = self.to_q(x).reshape(b, ln, self.n_heads, self.head_dim)
        k = self.to_k(x).reshape(b, ln, self.n_heads, self.head_dim)
        v = self.to_v(x).reshape(b, ln, self.n_heads, self.head_dim)
        if self.norm_q is not None:
            q = self.norm_q(q)
            k = self.norm_k(k)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        # (B, L, H, D) -> (B, H, L, D) for SDPA
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        scale = self.head_dim**-0.5
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        o = o.transpose(0, 2, 1, 3).reshape(b, ln, self.n_heads * self.head_dim)
        return self.to_out[0](o)


class ZImageFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(_silu(self.w1(x)) * self.w3(x))


class ZImageBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, eps: float, qk_norm: bool, modulation: bool):
        super().__init__()
        self.modulation = modulation
        self.attention = ZImageAttention(dim, n_heads, qk_norm, eps)
        self.feed_forward = ZImageFeedForward(dim, _swiglu_hidden(dim))
        self.attention_norm1 = nn.RMSNorm(dim, eps=eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=eps)
        if modulation:
            # diffusers: Sequential(Linear(256, 4*dim)); list -> key `adaLN_modulation.0`
            self.adaLN_modulation = [nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)]

    def __call__(
        self, x: mx.array, cos: mx.array, sin: mx.array, adaln: mx.array | None
    ) -> mx.array:
        if self.modulation:
            mod = self.adaLN_modulation[0](adaln)  # (B, 4*dim)
            scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(mod[:, None, :], 4, axis=2)
            gate_msa, gate_mlp = mx.tanh(gate_msa), mx.tanh(gate_mlp)
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
            attn = self.attention(self.attention_norm1(x) * scale_msa, cos, sin)
            x = x + gate_msa * self.attention_norm2(attn)
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        else:
            attn = self.attention(self.attention_norm1(x), cos, sin)
            x = x + self.attention_norm2(attn)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))
        return x


class ZImageTimestepEmbedder(nn.Module):
    def __init__(self, out_size: int, mid_size: int, freq_size: int = 256):
        super().__init__()
        self.freq_size = freq_size
        self.l1 = nn.Linear(freq_size, mid_size, bias=True)  # diffusers mlp.0
        self.l2 = nn.Linear(mid_size, out_size, bias=True)  # diffusers mlp.2

    def __call__(self, t: mx.array) -> mx.array:
        half = self.freq_size // 2
        freqs = mx.exp(-math.log(10000) * mx.arange(0, half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        return self.l2(_silu(self.l1(emb)))


class ZImageCapEmbedder(nn.Module):
    def __init__(self, cap_feat_dim: int, dim: int, eps: float):
        super().__init__()
        self.norm = nn.RMSNorm(cap_feat_dim, eps=eps)  # diffusers cap_embedder.0
        self.proj = nn.Linear(cap_feat_dim, dim, bias=True)  # diffusers cap_embedder.1

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(self.norm(x))


class ZImageFinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, eps=1e-6, affine=False)
        self.linear = nn.Linear(dim, out_dim, bias=True)
        self.adaLN_proj = nn.Linear(min(dim, ADALN_EMBED_DIM), dim, bias=True)  # adaLN_modulation.1

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        scale = 1.0 + self.adaLN_proj(_silu(c))[:, None, :]
        return self.linear(self.norm_final(x) * scale)


class ZImageTransformer(nn.Module):
    """Z-Image S3-DiT. ``__call__(latent (1,C,H,W), t, cap_feats (L, cap_dim)) -> (C,H,W)``."""

    def __init__(
        self,
        in_channels: int = 16,
        dim: int = 3840,
        n_layers: int = 30,
        n_refiner_layers: int = 2,
        n_heads: int = 30,
        norm_eps: float = 1e-5,
        qk_norm: bool = True,
        cap_feat_dim: int = 2560,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: tuple[int, ...] = (32, 48, 48),
        patch_size: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.dim = dim
        self.n_heads = n_heads
        self.t_scale = t_scale
        self.rope_theta = rope_theta
        self.axes_dims = list(axes_dims)
        self.patch_size = patch_size
        patch_dim = patch_size * patch_size * in_channels
        assert dim // n_heads == sum(axes_dims), "head_dim must equal sum(axes_dims)"

        self.x_embedder = nn.Linear(patch_dim, dim, bias=True)  # diffusers all_x_embedder["2-1"]
        self.cap_embedder = ZImageCapEmbedder(cap_feat_dim, dim, norm_eps)
        self.t_embedder = ZImageTimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.x_pad_token = mx.zeros((1, dim))
        self.cap_pad_token = mx.zeros((1, dim))
        def block(mod: bool) -> ZImageBlock:
            return ZImageBlock(dim, n_heads, norm_eps, qk_norm, modulation=mod)

        self.noise_refiner = [block(True) for _ in range(n_refiner_layers)]
        self.context_refiner = [block(False) for _ in range(n_refiner_layers)]
        self.layers = [block(True) for _ in range(n_layers)]
        self.final_layer = ZImageFinalLayer(dim, patch_dim)  # diffusers all_final_layer["2-1"]
        # caching (configured by the engine): static caption cache + first-block cache
        self.cache_threshold = 0.0
        self._cached_caption = None
        self._cached_caption_key = None
        self._prev_first_block_output = None
        self._prev_output = None
        self.computed_count = 0
        self.skipped_count = 0

    def reset_cache(self) -> None:
        self._cached_caption = None
        self._cached_caption_key = None
        self._prev_first_block_output = None
        self._prev_output = None
        self.computed_count = 0
        self.skipped_count = 0

    def _patchify(self, image: mx.array) -> tuple[mx.array, int, int]:
        """(C, H, W) -> (H_t*W_t, patch_dim) with patch feature order (pH, pW, C)."""
        p = self.patch_size
        c, h, w = image.shape
        ht, wt = h // p, w // p
        x = image.reshape(c, ht, p, wt, p)  # C, Ht, pH, Wt, pW
        x = x.transpose(1, 3, 2, 4, 0)  # Ht, Wt, pH, pW, C
        return x.reshape(ht * wt, p * p * c), ht, wt

    def _unpatchify(self, tokens: mx.array, ht: int, wt: int) -> mx.array:
        """(H_t*W_t, patch_dim) -> (C, H, W)."""
        p = self.patch_size
        c = self.out_channels
        x = tokens.reshape(ht, wt, p, p, c)  # Ht, Wt, pH, pW, C
        x = x.transpose(4, 0, 2, 1, 3)  # C, Ht, pH, Wt, pW
        return x.reshape(c, ht * p, wt * p)

    def __call__(self, latent: mx.array, t: mx.array, cap_feats: mx.array) -> mx.array:
        # latent (1, C, H, W); t (1,) in [0,1]; cap_feats (cap_len, cap_dim)
        image = latent[0]  # (C, H, W)
        adaln = self.t_embedder(t * self.t_scale)  # (1, 256)

        # --- image tokens + RoPE positions ---
        img_tokens, ht, wt = self._patchify(image)  # (Ht*Wt, patch_dim)
        x = self.x_embedder(img_tokens)[None]  # (1, Limg, dim)
        cap_len = cap_feats.shape[0]
        cap_padded = ((cap_len + SEQ_MULTI_OF - 1) // SEQ_MULTI_OF) * SEQ_MULTI_OF
        img_axis0 = cap_padded + 1
        hh = mx.repeat(mx.arange(ht), wt)  # h index per token (h outer, w inner)
        ww = mx.tile(mx.arange(wt), ht)
        a0 = mx.full((ht * wt,), img_axis0, dtype=mx.int32)
        img_pos = mx.stack([a0, hh.astype(mx.int32), ww.astype(mx.int32)], axis=-1)
        img_cos, img_sin = build_rope_cos_sin(img_pos, self.axes_dims, self.rope_theta)

        def r(c):  # (L, D/2) -> (1, L, 1, D/2) for broadcast over heads
            return c[None, :, None, :]

        # --- caption RoPE positions (fixed per generation) ---
        zeros = mx.zeros((cap_len,), mx.int32)
        cap_pos = mx.stack([mx.arange(1, cap_len + 1, dtype=mx.int32), zeros, zeros], axis=-1)
        cap_cos, cap_sin = build_rope_cos_sin(cap_pos, self.axes_dims, self.rope_theta)

        # caption embed + context-refine is timestep-independent -> cache once per generation
        if self._cached_caption is not None and self._cached_caption_key is cap_feats:
            cap = self._cached_caption
        else:
            cap = self.cap_embedder(cap_feats)[None]  # (1, cap_len, dim)
            for blk in self.context_refiner:
                cap = blk(cap, r(cap_cos), r(cap_sin), None)
            mx.eval(cap)
            self._cached_caption = cap
            self._cached_caption_key = cap_feats

        # image refine is timestep-dependent (not cached)
        for blk in self.noise_refiner:
            x = blk(x, r(img_cos), r(img_sin), adaln)

        unified = mx.concatenate([x, cap], axis=1)
        ucos = r(mx.concatenate([img_cos, cap_cos], axis=0))
        usin = r(mx.concatenate([img_sin, cap_sin], axis=0))

        # first-block cache: run layer 0; skip the rest if its step-to-step change is tiny
        unified = self.layers[0](unified, ucos, usin, adaln)
        if self.cache_threshold > 0.0:
            prev = self._prev_first_block_output
            if prev is not None and self._prev_output is not None:
                diff = mx.mean(mx.abs(unified - prev)) / mx.mean(mx.abs(prev))
                if diff.item() < self.cache_threshold:
                    self.skipped_count += 1
                    return self._prev_output
            self._prev_first_block_output = unified

        self.computed_count += 1
        for blk in self.layers[1:]:
            unified = blk(unified, ucos, usin, adaln)
        unified = self.final_layer(unified, adaln)  # (1, Lunified, patch_dim)
        out = self._unpatchify(unified[0, : ht * wt], ht, wt)  # drop caption tail -> (C, H, W)
        if self.cache_threshold > 0.0:
            self._prev_output = out
        return out
