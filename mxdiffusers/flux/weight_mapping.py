"""HF checkpoint key -> mxalloy module param-path remapping for klein.

Our module attribute names mirror the FLUX.2 reference, so the remap is small and purely
string-based (kept mlx-free so it is trivially testable):

- text encoder: Qwen3 keys live under ``model.``; strip it. Skip non-model keys (lm_head).
- transformer: drop ``timestep_embedder.``; collapse ``attn.to_out.0`` -> ``attn.to_out``.
- VAE (decode-only): keep ``decoder.`` / ``post_quant_conv.`` / ``bn.`` keys; collapse
  ``to_out.0`` -> ``to_out``. Encoder / quant_conv keys are dropped.

Conv-weight *layout* conversion (PyTorch [out,in,kh,kw] -> mlx [out,kh,kw,in]) is applied
by the loader, since it needs mlx; here we only rewrite key strings. A returned ``None``
means "skip this checkpoint key".
"""

from __future__ import annotations

_VAE_DECODE_PREFIXES = ("decoder.", "post_quant_conv.", "bn.")


def remap_text_encoder_key(key: str) -> str | None:
    if not key.startswith("model."):
        return None
    return key[len("model.") :]


def remap_transformer_key(key: str) -> str:
    key = key.replace(".timestep_embedder.", ".")
    return key.replace(".to_out.0.", ".to_out.")


def remap_vae_decode_key(key: str) -> str | None:
    if not key.startswith(_VAE_DECODE_PREFIXES):
        return None
    return key.replace(".to_out.0.", ".to_out.")
