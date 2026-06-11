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

__all__ = [
    "ActivationOption",
    "AppleSiliconDevice",
    "ComponentSpec",
    "DeviceProfile",
    "ExecutionStrategy",
    "WorkloadSpec",
    "detect_device",
    "detect_device_profile",
    "estimate_peak_gb",
    "plan_execution",
]
