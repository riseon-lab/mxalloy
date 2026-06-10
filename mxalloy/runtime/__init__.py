"""Runtime planning and execution helpers."""

from mxalloy.runtime.device import (
    AppleSiliconDevice,
    DeviceProfile,
    detect_device,
    detect_device_profile,
)
from mxalloy.runtime.planner import (
    ActivationOption,
    ComponentSpec,
    ExecutionStrategy,
    WorkloadSpec,
    estimate_peak_gb,
    plan_execution,
)
from mxalloy.runtime.scheduler import ExecutionStep, RuntimeSchedule

__all__ = [
    "ActivationOption",
    "AppleSiliconDevice",
    "ComponentSpec",
    "DeviceProfile",
    "ExecutionStep",
    "ExecutionStrategy",
    "RuntimeSchedule",
    "WorkloadSpec",
    "detect_device",
    "detect_device_profile",
    "estimate_peak_gb",
    "plan_execution",
]
