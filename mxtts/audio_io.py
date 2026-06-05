"""Small WAV writer used by audio pipelines without importing torchaudio at module import."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np


def save_wav(audio: Any, sample_rate: int, path: str | Path) -> Path:
    """Save mono or stereo float audio to a 16-bit PCM WAV file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    samples = _as_numpy(audio)
    if samples.ndim == 1:
        samples = samples[:, None]
    elif samples.ndim == 2 and samples.shape[0] <= 8 and samples.shape[1] > samples.shape[0]:
        # Common tensor layout from audio libraries is channels x samples.
        samples = samples.T
    elif samples.ndim != 2:
        raise ValueError(f"expected 1D or 2D audio, got shape {samples.shape}")

    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * np.iinfo(np.int16).max).astype("<i2", copy=False)

    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(int(pcm.shape[1]))
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())
    return output


def _as_numpy(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach()
    if hasattr(audio, "float"):
        audio = audio.float()
    if hasattr(audio, "cpu"):
        audio = audio.cpu()
    if hasattr(audio, "numpy"):
        audio = audio.numpy()
    return np.asarray(audio, dtype=np.float32)
