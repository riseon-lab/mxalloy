"""Generate an image from text with FLUX.2-klein on Apple Silicon (MLX).

Requires the MLX extra and a local FLUX.2-klein-4B checkpoint in the Hugging Face cache::

    pip install "mxalloy[mlx]"
    huggingface-cli download black-forest-labs/FLUX.2-klein-4B
    python examples/flux_text_to_image.py --prompt "a brushed alloy sculpture, studio light"

The 4-bit quantized load keeps the model resident on ~18 GB of unified memory, and the VAE
decode is tiled so larger images stay within budget. This script drives the current FLUX
engine directly; a model-agnostic ``mxalloy.loader(...)`` front door is the next milestone.
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLUX.2-klein text-to-image via mxalloy (MLX).")
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", default="out.png")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument(
        "--bits",
        type=int,
        choices=[4, 8],
        default=4,
        help="weight quantization for the transformer + text encoder (VAE stays bf16)",
    )
    p.add_argument(
        "--lora",
        action="append",
        default=[],
        metavar="PATH[:STRENGTH]",
        help="LoRA safetensors to apply (repeatable); optional ':strength' suffix",
    )
    return p.parse_args()


def _parse_lora(spec: str) -> tuple[str, float]:
    path, _, strength = spec.partition(":")
    return path, float(strength) if strength else 1.0


def main() -> None:
    args = parse_args()
    # Imported lazily so `--help` works without MLX installed.
    from mxdiffusers.flux.engine import Flux2KleinEngine

    engine = Flux2KleinEngine(quantize_bits=args.bits)
    if args.lora:
        summary = engine.set_loras([_parse_lora(s) for s in args.lora])
        print(f"LoRA: applied={summary['applied']} skipped={len(summary['skipped'])}")

    image = engine.generate(
        args.prompt,
        seed=args.seed,
        steps=args.steps,
        height=args.height,
        width=args.width,
    )
    image.save(args.out)
    print(f"saved {args.width}x{args.height} -> {args.out}")


if __name__ == "__main__":
    main()
