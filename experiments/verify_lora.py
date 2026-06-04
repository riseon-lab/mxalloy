"""Verify runtime LoRA + hot-swap on the quantized klein-4B base (headless).

base -> apply trpfrog -> hot-swap to zoom -> clear, asserting each LoRA changes the output
and that clearing restores the base bit-for-bit (the wrapper with no deltas == base).

    PYTHONPATH=. .venv/bin/python experiments/verify_lora.py
"""

from __future__ import annotations

import glob
import os

import numpy as np
from huggingface_hub import snapshot_download

from mxdiffusers.flux.engine import Flux2KleinEngine
from mxdiffusers.flux.lora import apply_loras, clear_loras, load_lora_file

SEED, PROMPT, H, W = 42, "a brushed alloy sculpture under studio light", 512, 512
LORAS = {
    "trpfrog": "Prgckwb/trpfrog-lora-flux2-klein-4b-v1",
    "zoom": "fal/flux-2-klein-4B-zoom-lora",
}


def lora_path(repo: str) -> str:
    d = snapshot_download(repo, local_files_only=True)
    return sorted(glob.glob(os.path.join(d, "**", "*.safetensors"), recursive=True))[0]


def gen(engine) -> np.ndarray:
    return np.asarray(engine.generate(PROMPT, seed=SEED, steps=4, height=H, width=W)).astype(np.int16)


def md(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def main() -> None:
    paths = {k: lora_path(v) for k, v in LORAS.items()}
    engine = Flux2KleinEngine(quantize_bits=4)

    base = gen(engine)

    s = apply_loras(engine.transformer, [(load_lora_file(paths["trpfrog"]), 1.0)])
    print(f"trpfrog applied: {s}")
    trp = gen(engine)

    s = apply_loras(engine.transformer, [(load_lora_file(paths["zoom"]), 1.0)])  # hot-swap
    print(f"zoom applied (hot-swap): {s}")
    zoom = gen(engine)

    clear_loras(engine.transformer)
    cleared = gen(engine)

    for name, im in [("base", base), ("trpfrog", trp), ("zoom", zoom), ("cleared", cleared)]:
        from PIL import Image
        Image.fromarray(im.astype(np.uint8)).save(f"experiments/lora_{name}.png")

    print(f"\ntrpfrog vs base : {md(trp, base):6.2f}/255   (expect > 0: LoRA changes output)")
    print(f"zoom    vs base : {md(zoom, base):6.2f}/255   (expect > 0)")
    print(f"trpfrog vs zoom : {md(trp, zoom):6.2f}/255   (expect > 0: hot-swap is real)")
    print(f"cleared vs base : {md(cleared, base):6.2f}/255   (expect ~0: clean unload)")


if __name__ == "__main__":
    main()
