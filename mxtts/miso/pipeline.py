"""MXMisoTTSPipeline - Miso TTS 8B adapter.

This first pass is intentionally hybrid: it gives Alloy a stable speech pipeline surface while
delegating the released PyTorch/Moshi implementation to the upstream MisoTTS package. The native
mxalloy path will replace the backbone/codebook generator once the tensor mapping and quantized
KV decode are in place.
"""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mxtts.pipeline import MXAudioResult, MXTTSPipeline

DEFAULT_MISO_TTS_MODEL = "MisoLabs/MisoTTS"


class MXMisoTTSUnavailable(ImportError):
    """Raised when the upstream MisoTTS runtime is not importable."""


@dataclass(frozen=True, slots=True)
class MisoTTSRunConfig:
    speaker: int = 0
    max_audio_length_ms: float = 10_000
    temperature: float = 0.9
    topk: int = 50


class MXMisoTTSPipeline(MXTTSPipeline):
    """Miso TTS 8B text-to-speech pipeline.

    ``backend="upstream"`` loads the released MisoTTS PyTorch/Moshi implementation. The native
    MLX/mxalloy backend is reserved and will be used for quantized streaming load once ported.
    """

    family = "miso"

    def __init__(
        self,
        *,
        model_id: str,
        backend: str,
        generator: Any,
        device: str,
        dtype: str,
        source_path: Path | None,
    ) -> None:
        self.model_id = model_id
        self.backend = backend
        self.device = device
        self.dtype = dtype
        self.source_path = source_path
        self._generator = generator
        self.sample_rate = int(generator.sample_rate)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str | None = None,
        *,
        backend: str = "upstream",
        source_path: str | Path | None = None,
        device: str | None = None,
        dtype: str = "bfloat16",
        quantize_bits: int | None = None,
        **kwargs: Any,
    ) -> MXMisoTTSPipeline:
        model = model_id or DEFAULT_MISO_TTS_MODEL
        backend = backend.lower()
        if backend in {"mlx", "mxalloy", "native"}:
            raise NotImplementedError(
                "Native MXMisoTTS loading is not implemented yet. Use backend='upstream' for "
                "the first audio path while the mxalloy tensor mapping lands."
            )
        if backend != "upstream":
            raise ValueError(f"Unsupported MisoTTS backend: {backend!r}")
        if quantize_bits is not None:
            raise NotImplementedError(
                "quantize_bits is reserved for the native mxalloy backend; the upstream "
                "PyTorch/Moshi backend loads the checkpoint as released by Miso Labs."
            )

        upstream = _import_upstream(source_path)
        torch = importlib.import_module("torch")
        resolved_device = device or _default_device(torch)
        resolved_dtype = _resolve_dtype(torch, dtype)
        generator = upstream.load_miso_8b(
            device=resolved_device,
            model_path_or_repo_id=model,
            dtype=resolved_dtype,
        )
        return cls(
            model_id=model,
            backend=backend,
            generator=generator,
            device=resolved_device,
            dtype=dtype,
            source_path=Path(source_path).expanduser().resolve() if source_path else None,
        )

    def __call__(
        self,
        text: str,
        *,
        speaker: int = 0,
        context: list[Any] | None = None,
        max_audio_length_ms: float = 10_000,
        temperature: float = 0.9,
        topk: int = 50,
        **kwargs: Any,
    ) -> MXAudioResult:
        if not text.strip():
            raise ValueError("text must not be empty")
        audio = self._generator.generate(
            text=text,
            speaker=int(speaker),
            context=list(context or []),
            max_audio_length_ms=float(max_audio_length_ms),
            temperature=float(temperature),
            topk=int(topk),
        )
        return MXAudioResult(
            audio=audio,
            sample_rate=self.sample_rate,
            metadata={
                "model_id": self.model_id,
                "backend": self.backend,
                "device": self.device,
                "dtype": self.dtype,
                "speaker": int(speaker),
                "max_audio_length_ms": float(max_audio_length_ms),
                "temperature": float(temperature),
                "topk": int(topk),
            },
        )


def _import_upstream(source_path: str | Path | None) -> Any:
    try:
        with _prepend_sys_path(source_path):
            module = importlib.import_module("generator")
    except Exception as exc:
        location = f" from {source_path}" if source_path else ""
        raise MXMisoTTSUnavailable(
            "Could not import the upstream MisoTTS generator"
            f"{location}. Clone https://github.com/MisoLabsAI/MisoTTS and pass "
            "source_path='/path/to/MisoTTS', or install that project in this environment."
        ) from exc
    if not hasattr(module, "load_miso_8b"):
        raise MXMisoTTSUnavailable("Imported generator module does not expose load_miso_8b")
    return module


@contextmanager
def _prepend_sys_path(source_path: str | Path | None):
    if source_path is None:
        yield
        return
    path = str(Path(source_path).expanduser().resolve())
    sys.path.insert(0, path)
    try:
        yield
    finally:
        with suppress(ValueError):
            sys.path.remove(path)


def _default_device(torch: Any) -> str:
    if getattr(torch.cuda, "is_available", lambda: False)():
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and getattr(mps, "is_available", lambda: False)():
        return "mps"
    return "cpu"


def _resolve_dtype(torch: Any, dtype: str) -> Any:
    normalized = dtype.lower().replace("-", "")
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype for MisoTTS upstream backend: {dtype!r}")
    return mapping[normalized]
