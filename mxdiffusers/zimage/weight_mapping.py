"""diffusers Z-Image checkpoint key -> mxdiffusers.zimage module param-path remap.

Our transformer attribute names mirror the diffusers state_dict, so block-internal keys
(``layers.N.*``, ``noise_refiner.N.*``, ``context_refiner.N.*``) are identity; only a handful
of top-level embedders/final-layer keys are renamed. The text encoder is the same Qwen3 as
klein (strip ``model.``); the VAE is a stock AutoencoderKL decoder (keep ``decoder.``, collapse
``to_out.0`` -> ``to_out``). A returned ``None`` skips the key. Conv layout conversion is the
loader's job. mlx-free / pure string ops.
"""

from __future__ import annotations

_TRANSFORMER_IDENTITY = ("layers.", "noise_refiner.", "context_refiner.")
_TRANSFORMER_RENAME = {
    "all_x_embedder.2-1.": "x_embedder.",
    "cap_embedder.0.": "cap_embedder.norm.",
    "cap_embedder.1.": "cap_embedder.proj.",
    "t_embedder.mlp.0.": "t_embedder.l1.",
    "t_embedder.mlp.2.": "t_embedder.l2.",
    "all_final_layer.2-1.linear.": "final_layer.linear.",
    "all_final_layer.2-1.adaLN_modulation.1.": "final_layer.adaLN_proj.",
}


def remap_zimage_transformer_key(key: str) -> str | None:
    if key in ("x_pad_token", "cap_pad_token"):
        return key
    if key.startswith(_TRANSFORMER_IDENTITY):
        return key  # block-internal keys already match (incl. to_out.0, adaLN_modulation.0)
    for src, dst in _TRANSFORMER_RENAME.items():
        if key.startswith(src):
            return dst + key[len(src) :]
    return None  # drop unknown keys (e.g. siglip_* on the Omni variant)


def remap_zimage_text_encoder_key(key: str) -> str | None:
    if not key.startswith("model."):
        return None  # drop lm_head etc.
    return key[len("model.") :]


def remap_zimage_vae_key(key: str) -> str | None:
    if not key.startswith("decoder."):
        return None  # decode-only: drop encoder / quant convs
    return key.replace(".to_out.0.", ".to_out.")
