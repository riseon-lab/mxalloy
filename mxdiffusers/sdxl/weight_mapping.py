"""SDXL checkpoint key -> mxdiffusers.sdxl module param-path remap.

Module attribute names mirror the diffusers state_dict, so the remap is identity plus
filters: the VAE loads decode-only (drop ``encoder.*`` / ``quant_conv.*``) and the text
encoders drop the legacy ``position_ids`` buffer some exports carry. A returned ``None``
skips the key. Conv layout conversion is the loader's job. mlx-free / pure string ops.
"""

from __future__ import annotations


def remap_sdxl_unet_key(key: str) -> str | None:
    return key  # exact mirror


def remap_sdxl_vae_key(key: str) -> str | None:
    if not (key.startswith("decoder.") or key.startswith("post_quant_conv.")):
        return None  # decode-only: drop encoder / quant_conv
    return key


def remap_sdxl_text_encoder_key(key: str) -> str | None:
    if key.endswith("position_ids"):
        return None  # legacy buffer, not a parameter
    return key
