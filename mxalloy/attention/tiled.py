"""Planning objects for tiled attention execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TiledAttentionPlan:
    query_tile: int = 128
    key_tile: int = 128
    head_tile: int = 1

    def validate(self) -> None:
        if self.query_tile <= 0 or self.key_tile <= 0 or self.head_tile <= 0:
            raise ValueError("Attention tile sizes must be positive.")

