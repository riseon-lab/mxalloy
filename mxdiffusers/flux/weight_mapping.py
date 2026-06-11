"""FLUX.2-klein checkpoint key -> mxdiffusers.flux module param-path remap.

Module attribute names mirror the diffusers-format checkpoint, so the remap is identity plus
filters: the VAE loads decode-only (drop ``encoder.*`` / ``quant_conv.*``, keep the ``bn``
running stats, drop the useless ``num_batches_tracked`` scalar) and the text encoder drops
``lm_head`` if a checkpoint carries one (klein ties embeddings). A returned ``None`` skips
the key. Conv layout conversion is the loader's job. mlx-free / pure string ops.
"""

from __future__ import annotations


def remap_transformer_key(key: str) -> str | None:
    return key  # exact mirror


def remap_text_encoder_key(key: str) -> str | None:
    if key.startswith("lm_head."):
        return None
    return key  # model.* mirrored


def remap_vae_decode_key(key: str) -> str | None:
    if key == "bn.num_batches_tracked":
        return None  # training counter, not used at inference
    if key.startswith(("decoder.", "post_quant_conv.", "bn.")):
        return key
    return None  # decode-only: drop encoder / quant_conv
