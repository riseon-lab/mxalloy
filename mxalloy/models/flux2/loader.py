"""Stream real klein checkpoint weights into resident modules (native MLX).

For each component, streams its safetensors one tensor at a time, remaps the HF key to our
module's param path, transposes 4D conv weights from PyTorch [out,in,kh,kw] to mlx
[out,kh,kw,in], and updates the (already-built, resident) module — freeing each source
tensor as it goes. Checkpoint keys with no matching module param are skipped.

With ``quantize_bits`` set, 2D weights are stream-quantized into nn.quantize'd layers
(the memory win); otherwise they load as bf16. INTERNAL: requires mlx.
"""

from __future__ import annotations

import glob
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten


def find_klein_model_dir() -> str:
    pattern = str(
        Path.home()
        / ".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B"
        / "snapshots/*"
    )
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        raise FileNotFoundError("FLUX.2-klein-4B not found in the Hugging Face cache")
    return dirs[-1]


def component_files(model_dir: str, component: str) -> list[str]:
    files = sorted(glob.glob(str(Path(model_dir) / component / "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors for component {component!r} in {model_dir}")
    return files


def load_into_module(
    module: nn.Module,
    files: list[str],
    remap: Callable[[str], str | None],
    *,
    quantize_bits: int | None = None,
    group_size: int = 64,
) -> set[str]:
    """Stream-load remapped real weights into ``module``, freeing per tensor.

    If ``quantize_bits`` is set the module is nn.quantize'd and 2D weights are quantized on
    the fly into (packed, scales, biases); otherwise weights load as bf16. 4D conv weights
    are transposed PyTorch -> mlx. Returns module param paths NOT populated (empty = full).
    """
    if quantize_bits is not None:
        nn.quantize(
            module,
            group_size=group_size,
            bits=quantize_bits,
            class_predicate=lambda _path, m: hasattr(m, "to_quantized"),
        )
    targets = {k for k, _ in tree_flatten(module.parameters())}
    quantized_bases = {k[: -len(".scales")] for k in targets if k.endswith(".scales")}
    updates: list[tuple[str, mx.array]] = []
    for path in files:
        weights = mx.load(path)  # lazy
        for hf_key in list(weights.keys()):
            module_key = remap(hf_key)
            if module_key is None:
                weights[hf_key] = None
                continue
            weight = weights[hf_key]
            if weight.ndim == 4:  # conv: PyTorch [out,in,kh,kw] -> mlx [out,kh,kw,in]
                weight = weight.transpose(0, 2, 3, 1)
            base = module_key[: -len(".weight")] if module_key.endswith(".weight") else None
            if base is not None and base in quantized_bases:
                packed, scales, biases = mx.quantize(
                    weight, group_size=group_size, bits=quantize_bits
                )
                mx.eval(packed, scales, biases)
                updates.append((f"{base}.weight", packed))
                updates.append((f"{base}.scales", scales))
                updates.append((f"{base}.biases", biases))
            elif module_key in targets:
                mx.eval(weight)
                updates.append((module_key, weight))
            weights[hf_key] = None
        del weights
    module.update(tree_unflatten(updates))
    mx.eval(module.parameters())
    return targets - {k for k, _ in updates}
