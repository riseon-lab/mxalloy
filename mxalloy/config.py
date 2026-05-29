"""Configuration objects shared across Alloy runtime components."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

QuantizationMode = Literal["none", "fp16", "int8", "int4"]
AttentionMode = Literal["mlx", "metal", "tiled"]


@dataclass(slots=True)
class QuantizationConfig:
    mode: QuantizationMode = "fp16"
    group_size: int = 64
    symmetric: bool = True
    pack_weights: bool = True


@dataclass(slots=True)
class RuntimeConfig:
    backend: Literal["mlx", "metal"] = "mlx"
    attention: AttentionMode = "mlx"
    memory_budget_gb: float | None = None
    enable_unified_memory_pressure_checks: bool = True


@dataclass(slots=True)
class AlloyConfig:
    model_id: str | None = None
    model_path: Path | None = None
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    lora_paths: list[Path] = field(default_factory=list)

