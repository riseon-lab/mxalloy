"""Runtime LoRA on the quantized base (no merge into 4-bit weights), hot-swappable.

A LoRA-adapted Linear computes ``y = base(x) + Σ scaleᵢ · (x @ Aᵢᵀ) @ Bᵢᵀ`` with the base
weights left quantized -- so a LoRA is applied at runtime and swapped without reloading the
model. Handles the diffusion_model / ComfyUI-BFL key convention (the common klein LoRA
format), including the fused-QKV -> split ``to_q/to_k/to_v`` mapping onto our diffusers-named
modules. Other conventions (diffusers, kohya) are TODO.

INTERNAL: requires mlx.
"""

from __future__ import annotations

import re

import mlx.core as mx
from mlx import nn

PRECISION = mx.bfloat16


class LoRALinear(nn.Module):
    """Wraps a (quantized) Linear, adding zero or more low-rank deltas at call time."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        # Held off the parameter tree (we never train/optimise these): plain attribute.
        object.__setattr__(self, "_loras", [])  # list of (A (r,in), B (out,r), scale)

    def add(self, a: mx.array, b: mx.array, scale: float) -> None:
        self._loras.append((a.astype(PRECISION), b.astype(PRECISION), float(scale)))

    def clear(self) -> None:
        self._loras.clear()

    def __call__(self, x: mx.array) -> mx.array:
        y = self.base(x)
        for a, b, scale in self._loras:
            y = y + scale * ((x @ a.T) @ b.T)
        return y


def _get(root, path: str):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _set(root, path: str, val) -> None:
    parts = path.split(".")
    obj = root
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = val
    else:
        setattr(obj, last, val)


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


def _wrap(transformer, path: str) -> LoRALinear:
    reg = getattr(transformer, "_lora_wrappers", None)
    if reg is None:
        reg = {}
        object.__setattr__(transformer, "_lora_wrappers", reg)
    if path in reg:
        return reg[path]
    base = _get(transformer, path)
    w = base if isinstance(base, LoRALinear) else LoRALinear(base)
    if not isinstance(base, LoRALinear):
        _set(transformer, path, w)
    reg[path] = w
    return w


def clear_loras(transformer) -> None:
    for w in getattr(transformer, "_lora_wrappers", {}).values():
        w.clear()


def _group(state: dict) -> dict:
    """state -> {base_path: {'A':arr,'B':arr,'alpha':float|None}} (diffusion_model convention)."""
    out: dict = {}
    for k, v in state.items():
        if k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight"):
            base = k.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
            out.setdefault(base, {})["A" if ".lora_A." in k else "B"] = v
        elif k.endswith(".alpha"):
            out.setdefault(k[: -len(".alpha")], {})["alpha"] = float(v)
    return out


def apply_loras(transformer, loras: list[tuple[dict, float]]) -> dict:
    """Replace all active LoRA deltas with ``loras`` (list of (state_dict, strength)).

    Hot-swap: the quantized base is untouched; only the per-wrapper delta list changes.
    Returns a summary: {'applied': n_layers, 'skipped': [unmapped base paths]}.
    """
    clear_loras(transformer)
    applied, skipped = 0, []
    for state, strength in loras:
        for base, ab in _group(state).items():
            if "A" not in ab or "B" not in ab:
                continue
            a, b = ab["A"], ab["B"]
            rank = a.shape[0]
            scale = strength * (ab["alpha"] / rank if ab.get("alpha") is not None else 1.0)
            p = base[len("diffusion_model."):] if base.startswith("diffusion_model.") else base
            targets = _bfl_targets(p)
            if not targets:
                skipped.append(p)
                continue
            for tpath, third in targets:
                bt = (
                    b
                    if third is None
                    else b[third * (b.shape[0] // 3): (third + 1) * (b.shape[0] // 3)]
                )
                _wrap(transformer, tpath).add(a, bt, scale)
                applied += 1
    return {"applied": applied, "skipped": sorted(set(skipped))}


def load_lora_file(path: str) -> dict:
    """Load a LoRA safetensors into a name->array dict."""
    return mx.load(path)
