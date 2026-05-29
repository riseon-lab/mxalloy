"""FLUX model adapter contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mxalloy.config import AlloyConfig


@dataclass(frozen=True, slots=True)
class FluxLoadRequest:
    model_id: str | None = None
    model_path: Path | None = None
    lora_paths: tuple[Path, ...] = field(default_factory=tuple)


class FluxAdapter:
    """Adapter boundary for FLUX-specific loading, LoRA merge, and inference."""

    def __init__(self, config: AlloyConfig) -> None:
        self.config = config

    def load(self, request: FluxLoadRequest) -> None:
        if request.model_id is None and request.model_path is None:
            raise ValueError("A FLUX model_id or model_path is required.")

    def generate(self, prompt: str) -> object:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        raise NotImplementedError("FLUX generation will be implemented in the prototype milestone.")

