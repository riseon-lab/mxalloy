"""Diffusers-style integration surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from mxalloy.config import AlloyConfig, QuantizationConfig


def enable_alloy(
    pipe: Any,
    *,
    quantization: Literal["none", "fp16", "int8", "int4"] = "fp16",
    loras: list[str | Path] | None = None,
) -> Any:
    """Attach Alloy configuration metadata to a compatible pipeline.

    The first working version will replace selected modules with MLX/Metal-backed
    implementations. Until then, this keeps the intended public API testable.
    """
    config = AlloyConfig(
        quantization=QuantizationConfig(mode=quantization),
        lora_paths=[Path(path) for path in loras or []],
    )
    pipe._mxalloy_config = config
    return pipe

