"""Apple Silicon device detection helpers."""

from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AppleSiliconDevice:
    machine: str
    processor: str
    is_apple_silicon: bool


def detect_device() -> AppleSiliconDevice:
    machine = platform.machine()
    processor = platform.processor()
    return AppleSiliconDevice(
        machine=machine,
        processor=processor,
        is_apple_silicon=machine == "arm64" and platform.system() == "Darwin",
    )

