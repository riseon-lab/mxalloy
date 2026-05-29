"""Registry for Metal kernel implementations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

KernelFactory = Callable[..., Any]


@dataclass(slots=True)
class KernelRegistry:
    _kernels: dict[str, KernelFactory] = field(default_factory=dict)

    def register(self, name: str, factory: KernelFactory) -> None:
        if not name:
            raise ValueError("Kernel name cannot be empty.")
        self._kernels[name] = factory

    def get(self, name: str) -> KernelFactory:
        try:
            return self._kernels[name]
        except KeyError as exc:
            raise KeyError(f"Kernel not registered: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._kernels))

