from __future__ import annotations

import wave

import numpy as np

from mxtts.pipeline import MXAudioResult


def test_audio_result_saves_wav(tmp_path) -> None:
    audio = np.zeros(2400, dtype=np.float32)
    result = MXAudioResult(audio=audio, sample_rate=24_000)
    path = result.save(tmp_path / "sample.wav")

    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 24_000
        assert handle.getnframes() == 2400
