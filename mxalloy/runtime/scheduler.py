"""Memory-aware runtime scheduling primitives."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ExecutionStep:
    name: str
    estimated_memory_mb: float | None = None


@dataclass(slots=True)
class RuntimeSchedule:
    steps: list[ExecutionStep] = field(default_factory=list)

    def add(self, step: ExecutionStep) -> None:
        self.steps.append(step)

    def max_step_memory_mb(self) -> float | None:
        estimates = [step.estimated_memory_mb for step in self.steps]
        known = [estimate for estimate in estimates if estimate is not None]
        if len(known) != len(estimates):
            return None
        return max(known, default=0.0)

