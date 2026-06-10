"""FLUX-family LoRA key mapping over the shared ``mxdiffusers.lora`` core.

Handles the diffusion_model / ComfyUI-BFL key convention (the common klein LoRA format),
including the fused-QKV -> split ``to_q/to_k/to_v`` mapping onto our diffusers-named modules.
Other conventions (diffusers/peft-prefixed, kohya-flattened) are TODO.

INTERNAL: requires mlx.
"""

from __future__ import annotations

import re
from typing import Any

from mlx import nn

from mxdiffusers.lora import LoRALinear, clear_loras, load_lora_file
from mxdiffusers.lora import apply_loras as _apply_loras

__all__ = ["LoRALinear", "apply_loras", "clear_loras", "load_lora_file"]


def _bfl_targets(p: str) -> list[tuple[str, int | None]]:
    """Map a diffusion_model base path -> [(our_module_path, qkv_third | None)].

    qkv_third in {0,1,2} means: take rows [third*out/3 : (third+1)*out/3] of the LoRA's B
    (the fused QKV splits into to_q/to_k/to_v, sharing A).
    """
    if m := re.match(r"double_blocks\.(\d+)\.(img|txt)_attn\.qkv$", p):
        i, s = m.group(1), m.group(2)
        names = (
            ["to_q", "to_k", "to_v"] if s == "img" else ["add_q_proj", "add_k_proj", "add_v_proj"]
        )
        return [(f"transformer_blocks.{i}.attn.{n}", k) for k, n in enumerate(names)]
    if m := re.match(r"double_blocks\.(\d+)\.(img|txt)_attn\.proj$", p):
        i, s = m.group(1), m.group(2)
        return [(f"transformer_blocks.{i}.attn.{'to_out' if s == 'img' else 'to_add_out'}", None)]
    if m := re.match(r"double_blocks\.(\d+)\.(img|txt)_mlp\.(\d)$", p):
        i, s, layer = m.group(1), m.group(2), m.group(3)
        ff = "ff" if s == "img" else "ff_context"
        return [
            (f"transformer_blocks.{i}.{ff}.{'linear_in' if layer == '0' else 'linear_out'}", None)
        ]
    if m := re.match(r"single_blocks\.(\d+)\.linear(\d)$", p):
        i, layer = m.group(1), m.group(2)
        sub = "to_qkv_mlp_proj" if layer == "1" else "to_out"
        return [(f"single_transformer_blocks.{i}.attn.{sub}", None)]
    if m := re.match(
        r"(double_stream_modulation_img|double_stream_modulation_txt|single_stream_modulation)\.lin$",
        p,
    ):
        return [(f"{m.group(1)}.linear", None)]
    if p == "img_in":
        return [("x_embedder", None)]
    if p == "txt_in":
        return [("context_embedder", None)]
    if p == "final_layer.linear":
        return [("proj_out", None)]
    if p == "time_in.in_layer":
        return [("time_guidance_embed.linear_1", None)]
    if p == "time_in.out_layer":
        return [("time_guidance_embed.linear_2", None)]
    return []  # unmapped -> skipped (reported by apply_loras)


def _targets_for_base(base: str) -> tuple[list[tuple[str, int | None]], str]:
    p = base[len("diffusion_model.") :] if base.startswith("diffusion_model.") else base
    return _bfl_targets(p), p


def apply_loras(transformer: nn.Module, loras: list[tuple[dict[str, Any], float]]) -> dict:
    """Replace all active LoRA deltas on the resident FLUX transformer (hot-swap)."""
    return _apply_loras(transformer, loras, targets_for_base=_targets_for_base)
