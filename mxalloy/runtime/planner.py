"""Adaptive memory planner for model execution.

The planner is intentionally model-agnostic: model families supply a ``WorkloadSpec`` with
measured or estimated memory for their components, and mxalloy chooses the best plan that fits
the detected machine budget.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from mxalloy.errors import ConfigurationError
from mxalloy.runtime.device import DeviceProfile

Precision = Literal["bf16", "int8", "int4"]
MemoryMode = Literal["resident", "staged", "survival"]

_PRECISION_LADDER: tuple[Precision, ...] = ("bf16", "int8", "int4")
# Estimated bytes/param for mlx affine quantization at group_size=64 with fp16 scales+biases
# ((bits + 4/64 * 32) / 8). Measured precision_memory_gb values bypass this estimate.
_BYTES_PER_PARAM = {
    "bf16": 2.0,
    "int8": 1.06,
    "int4": 0.56,
}


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """A model component's memory contribution.

    ``precision_memory_gb`` can hold measured values. If omitted, ``params`` is used with a
    conservative bytes-per-parameter estimate.
    """

    name: str
    params: int | None = None
    quantizable: bool = True
    precision_memory_gb: Mapping[Precision, float] = field(default_factory=dict)

    def memory_gb(self, precision: Precision) -> float:
        effective = precision if self.quantizable else "bf16"
        if effective in self.precision_memory_gb:
            return float(self.precision_memory_gb[effective])
        if self.params is None:
            raise ConfigurationError(
                f"Component {self.name!r} needs params or precision_memory_gb for planning"
            )
        return self.params * _BYTES_PER_PARAM[effective] / 1024**3


@dataclass(frozen=True, slots=True)
class ActivationOption:
    """Peak transient memory for a memory mode, usually VAE tile/workspace dominated."""

    memory_mode: MemoryMode
    activation_peak_gb: float
    vae_tile_latent: int | None = None


@dataclass(frozen=True, slots=True)
class WorkloadSpec:
    """A model family's planning inputs.

    ``activation_options`` must be ordered best-quality-first: the planner prefers earlier
    options and only falls back to later (smaller-peak) ones when earlier ones don't fit.
    """

    name: str
    components: tuple[ComponentSpec, ...]
    activation_options: tuple[ActivationOption, ...]
    default_steps: int
    overhead_gb: float = 0.0


@dataclass(frozen=True, slots=True)
class ExecutionStrategy:
    workload_name: str
    precision: Precision
    component_precisions: Mapping[str, Precision]
    memory_mode: MemoryMode
    vae_tile_latent: int | None
    steps: int
    estimated_peak_gb: float
    working_set_gb: float
    fits: bool
    reason: str
    warnings: tuple[str, ...] = ()

    @property
    def quant_bits(self) -> int | None:
        if self.precision == "int4":
            return 4
        if self.precision == "int8":
            return 8
        return None

    def to_payload(self) -> dict[str, object]:
        return {
            "workload_name": self.workload_name,
            "precision": self.precision,
            "component_precisions": dict(self.component_precisions),
            "memory_mode": self.memory_mode,
            "vae_tile_latent": self.vae_tile_latent,
            "steps": self.steps,
            "estimated_peak_gb": self.estimated_peak_gb,
            "working_set_gb": self.working_set_gb,
            "fits": self.fits,
            "reason": self.reason,
            "warnings": list(self.warnings),
        }


def estimate_peak_gb(
    workload: WorkloadSpec,
    precision: Precision,
    activation: ActivationOption,
) -> float:
    component_gb = sum(component.memory_gb(precision) for component in workload.components)
    return round(component_gb + activation.activation_peak_gb + workload.overhead_gb, 2)


def plan_execution(
    device: DeviceProfile,
    workload: WorkloadSpec,
    *,
    requested_precision: Precision | None = None,
    requested_memory_mode: MemoryMode | None = None,
) -> ExecutionStrategy:
    """Choose the best precision/tile plan that fits the device working set.

    The ordering is deliberate: keep the highest-quality memory mode first, then choose the
    highest precision that fits inside that mode. On 18 GB klein-class workloads this preserves
    the known-good ``int4/resident`` choice instead of dropping to a smaller VAE tile just to
    fit int8.
    """
    precisions = (requested_precision,) if requested_precision else _PRECISION_LADDER
    activations = _activation_candidates(workload, requested_memory_mode)
    candidates = [
        (
            estimate_peak_gb(workload, precision, activation),
            precision,
            activation,
        )
        for activation in activations
        for precision in precisions
    ]
    for estimated, precision, activation in candidates:
        if estimated <= device.working_set_gb:
            return _strategy(
                workload=workload,
                precision=precision,
                activation=activation,
                estimated=estimated,
                device=device,
                fits=True,
                reason=(
                    f"Selected {precision}/{activation.memory_mode}: estimated "
                    f"{estimated:.2f} GB <= working set {device.working_set_gb:.2f} GB"
                ),
            )

    estimated, precision, activation = min(candidates, key=lambda item: item[0])
    return _strategy(
        workload=workload,
        precision=precision,
        activation=activation,
        estimated=estimated,
        device=device,
        fits=False,
        reason=(
            f"No plan fits working set {device.working_set_gb:.2f} GB; smallest plan is "
            f"{precision}/{activation.memory_mode} at {estimated:.2f} GB"
        ),
        warnings=("estimated_peak_exceeds_working_set",),
    )


def _activation_candidates(
    workload: WorkloadSpec,
    requested_memory_mode: MemoryMode | None,
) -> tuple[ActivationOption, ...]:
    if requested_memory_mode is None:
        return workload.activation_options
    matches = tuple(
        activation
        for activation in workload.activation_options
        if activation.memory_mode == requested_memory_mode
    )
    if not matches:
        raise ConfigurationError(
            f"Workload {workload.name!r} does not support memory mode {requested_memory_mode!r}"
        )
    return matches


def _strategy(
    *,
    workload: WorkloadSpec,
    precision: Precision,
    activation: ActivationOption,
    estimated: float,
    device: DeviceProfile,
    fits: bool,
    reason: str,
    warnings: tuple[str, ...] = (),
) -> ExecutionStrategy:
    return ExecutionStrategy(
        workload_name=workload.name,
        precision=precision,
        component_precisions={
            component.name: precision if component.quantizable else "bf16"
            for component in workload.components
        },
        memory_mode=activation.memory_mode,
        vae_tile_latent=activation.vae_tile_latent,
        steps=workload.default_steps,
        estimated_peak_gb=estimated,
        working_set_gb=device.working_set_gb,
        fits=fits,
        reason=reason,
        warnings=warnings,
    )
