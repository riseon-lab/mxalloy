"""Apple Silicon device detection helpers."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

from mxalloy.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class AppleSiliconDevice:
    machine: str
    processor: str
    is_apple_silicon: bool


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    machine: str
    processor: str
    is_apple_silicon: bool
    total_memory_gb: float | None
    working_set_gb: float
    os_reserve_gb: float
    safety_margin_gb: float
    memory_budget_gb: float | None = None


def detect_device() -> AppleSiliconDevice:
    machine = platform.machine()
    processor = platform.processor()
    return AppleSiliconDevice(
        machine=machine,
        processor=processor,
        is_apple_silicon=machine == "arm64" and platform.system() == "Darwin",
    )


def detect_device_profile(
    *,
    memory_budget_gb: float | None = None,
    os_reserve_gb: float = 1.5,
    safety_margin_gb: float = 0.5,
) -> DeviceProfile:
    """Detect the machine and derive a conservative model working-set budget.

    ``memory_budget_gb`` is an optional cap for model planning. It represents the maximum model
    working set the caller wants to allow after OS/cache reserves have been considered.
    """
    if memory_budget_gb is not None and memory_budget_gb < 0:
        raise ConfigurationError(f"memory_budget_gb must be >= 0, got {memory_budget_gb}")
    device = detect_device()
    total = _system_memory_gb()
    if total is None:
        working = memory_budget_gb if memory_budget_gb is not None else 0.0
    else:
        physical_working = max(0.0, total - os_reserve_gb - safety_margin_gb)
        working = (
            min(physical_working, memory_budget_gb)
            if memory_budget_gb is not None
            else physical_working
        )
    return DeviceProfile(
        machine=device.machine,
        processor=device.processor,
        is_apple_silicon=device.is_apple_silicon,
        total_memory_gb=total,
        working_set_gb=round(working, 2),
        os_reserve_gb=os_reserve_gb,
        safety_margin_gb=safety_margin_gb,
        memory_budget_gb=memory_budget_gb,
    )


def _system_memory_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round((pages * page_size) / 1024**3, 1)
    except (AttributeError, OSError, ValueError):
        return None
