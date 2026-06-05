"""Base interfaces for mxalloy text-to-speech pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MXAudioResult:
    """Speech result: an in-memory audio tensor/array plus enough metadata to save it."""

    audio: Any
    sample_rate: int
    seed: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def save(self, path: str | Path) -> Path:
        """Write the audio as 16-bit PCM WAV and return the final path."""
        from mxtts.audio_io import save_wav

        return save_wav(self.audio, self.sample_rate, path)


class MXTTSPipeline:
    """Base text-to-speech pipeline.

    Concrete families should expose ``from_pretrained`` and ``__call__`` in the same spirit as
    ``mxdiffusers.MXPipeline``. The runtime contract is audio-specific: return an
    :class:`MXAudioResult`, not image data.
    """

    family: str = "base"

    @classmethod
    def from_pretrained(cls, model_id: str | None = None, **kwargs) -> MXTTSPipeline:
        raise NotImplementedError

    def __call__(self, text: str, **kwargs) -> MXAudioResult:
        raise NotImplementedError

    @staticmethod
    def device():
        """The detected Apple Silicon device profile (from the mxalloy runtime)."""
        from mxalloy.runtime import detect_device

        return detect_device()
