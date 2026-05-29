"""Prototype entry point for FLUX text-to-image generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from mxalloy.config import AlloyConfig, QuantizationConfig
from mxalloy.models.flux import FluxAdapter, FluxLoadRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FLUX inference through Alloy.")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--lora", action="append", type=Path, default=[])
    parser.add_argument("--quantization", choices=["fp16", "int8"], default="fp16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AlloyConfig(quantization=QuantizationConfig(mode=args.quantization))
    adapter = FluxAdapter(config)
    adapter.load(
        FluxLoadRequest(
            model_id=args.model_id,
            model_path=args.model_path,
            lora_paths=tuple(args.lora),
        )
    )
    adapter.generate(args.prompt)


if __name__ == "__main__":
    main()

