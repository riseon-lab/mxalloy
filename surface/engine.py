"""Engine adapter boundary for the local tester surface.

``RealPipelineEngine`` is model-agnostic: it routes a ``model_id`` to its registered
``mxdiffusers`` pipeline (``MXFluxPipeline``, ``MXZimagePipeline``, …), keeps exactly one
pipeline resident (loading a different model frees the previous — two 6B-class models won't
co-reside on 18 GB), and drives the pipeline's ``on_step`` hook for progress so the UI never
reimplements a model's denoise loop. ``MockEngine`` is the dependency-free stand-in.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import textwrap
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from mxalloy.runtime import WorkloadSpec, detect_device_profile, plan_execution

Emit = Callable[[str, str, dict[str, Any] | None], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class LoraSelection:
    id: str
    strength: float = 1.0
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    model_id: str
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 4
    guidance: float = 0.0
    seed: int | None = None
    quant: str = "int4"
    memory_mode: str = "resident"
    refs: list[str] = field(default_factory=list)
    loras: list[LoraSelection] = field(default_factory=list)


class MockEngine:
    """Small deterministic stand-in for the resident mxalloy engine."""

    def __init__(self) -> None:
        self.mode = "mock"
        self.loaded_model_id: str | None = None
        self.loaded_quant: str | None = None
        self.loaded_memory_mode: str | None = None
        self._active_gb = 0.0
        self._peak_gb = 0.0

    async def load(self, model_id: str, quant: str, memory_mode: str, emit: Emit) -> None:
        if (
            self.loaded_model_id == model_id
            and self.loaded_quant == quant
            and self.loaded_memory_mode == memory_mode
        ):
            await emit("load", "Model already warm", {"model_id": model_id})
            return

        await emit("load", f"Loading {model_id}", {"model_id": model_id, "quant": quant})
        for label, memory in (
            ("reading manifest", 0.4),
            ("streaming quantized weights", 3.1),
            ("warming graph", 4.6),
        ):
            self._active_gb = memory
            self._peak_gb = max(self._peak_gb, memory)
            await asyncio.sleep(0.25)
            await emit(
                "load",
                label,
                {"active_memory": f"{memory:.1f} GB", "memory": self.memory_snapshot()},
            )

        self.loaded_model_id = model_id
        self.loaded_quant = quant
        self.loaded_memory_mode = memory_mode
        await emit(
            "ready",
            "Mock engine ready",
            {
                "model_id": model_id,
                "quant": quant,
                "memory_mode": memory_mode,
                "memory": self.memory_snapshot(),
            },
        )

    async def generate(
        self,
        req: GenerationRequest,
        output_dir: Path,
        emit: Emit,
    ) -> dict[str, Any]:
        await self.load(req.model_id, req.quant, req.memory_mode, emit)
        await emit(
            "generate",
            "Generation started",
            {"steps": req.steps, "width": req.width, "height": req.height},
        )

        for step in range(1, max(1, req.steps) + 1):
            await asyncio.sleep(0.12)
            self._active_gb = 4.6 + min(0.6, step * 0.06)
            self._peak_gb = max(self._peak_gb, self._active_gb)
            await emit(
                "progress",
                f"Step {step}/{req.steps}",
                {
                    "step": step,
                    "steps": req.steps,
                    "active_memory": f"{self._active_gb:.1f} GB",
                    "memory": self.memory_snapshot(),
                },
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        seed = req.seed if req.seed is not None else _stable_seed(req.prompt)
        filename = f"mxalloy_mock_{int(time.time())}_{seed}.png"
        path = output_dir / filename
        _render_mock_image(req, path, seed)
        meta = {
            "model_id": req.model_id,
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "guidance": req.guidance,
            "seed": seed,
            "quant": req.quant,
            "memory_mode": req.memory_mode,
            "refs": req.refs,
            "loras": [asdict(item) for item in req.loras],
            "mock": True,
            "created_at": time.time(),
        }
        path.with_suffix(".json").write_text(_json_dumps(meta), encoding="utf-8")
        await emit("complete", "Image written", {"path": str(path), "seed": seed})
        return {"path": path, "meta": meta}

    def memory_snapshot(self) -> dict[str, Any]:
        return _memory_payload(
            active_gb=self._active_gb,
            peak_gb=self._peak_gb,
            cache_gb=0.0,
            available=True,
            source="mock",
        )


class RealPipelineEngine:
    """Model-agnostic adapter around the resident mxdiffusers pipelines.

    ``pipeline_registry`` maps a ``model_id`` to a ``"module:ClassName"`` ``MXPipeline``. The
    engine keeps one pipeline resident and swaps on model change.
    """

    def __init__(
        self,
        model_dir_resolver: Callable[[str], str],
        pipeline_registry: dict[str, str],
        strategy_registry: dict[str, WorkloadSpec] | None = None,
        lora_resolver: Callable[[str], str] | None = None,
    ) -> None:
        self.mode = "real"
        self.loaded_model_id: str | None = None
        self.loaded_quant: str | None = None
        self.loaded_memory_mode: str | None = None
        self.last_strategy: dict[str, Any] | None = None
        self._pipe: Any | None = None
        self._model_dir_resolver = model_dir_resolver
        self._lora_resolver = lora_resolver or (lambda lora_id: lora_id)
        self._registry = pipeline_registry
        self._strategies = strategy_registry or {}
        self._active_lora_key: tuple[tuple[str, str, float], ...] = ()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mxalloy-surface")

    async def load(self, model_id: str, quant: str, memory_mode: str, emit: Emit) -> None:
        strategy, resolved_quant, resolved_memory_mode = self._resolve_strategy(
            model_id, quant, memory_mode
        )
        if (
            self._pipe is not None
            and self.loaded_model_id == model_id
            and self.loaded_quant == resolved_quant
            and self.loaded_memory_mode == resolved_memory_mode
        ):
            await emit(
                "ready",
                "Model already warm",
                {
                    "model_id": model_id,
                    "quant": resolved_quant,
                    "memory_mode": resolved_memory_mode,
                    "strategy": strategy,
                    "memory": self.memory_snapshot(),
                },
            )
            return
        if model_id not in self._registry:
            raise ValueError(f"No pipeline registered for model {model_id!r}")

        quant_bits = _quant_bits(resolved_quant)
        tile_latent = _tile_latent(resolved_memory_mode)
        model_dir = self._model_dir_resolver(model_id)
        await emit(
            "load",
            f"Loading {model_id}",
            {
                "model_id": model_id,
                "quant": resolved_quant,
                "memory_mode": resolved_memory_mode,
                "requested_quant": quant,
                "requested_memory_mode": memory_mode,
                "model_dir": model_dir,
                "strategy": strategy,
                "memory": self.memory_snapshot(),
            },
        )
        await self._run(self._dispose)
        await self._run(self._build, model_id, model_dir, quant_bits, tile_latent)
        self.loaded_model_id = model_id
        self.loaded_quant = resolved_quant
        self.loaded_memory_mode = resolved_memory_mode
        self.last_strategy = strategy
        await emit(
            "ready",
            "Pipeline ready",
            {
                "model_id": model_id,
                "quant": resolved_quant,
                "memory_mode": resolved_memory_mode,
                "requested_quant": quant,
                "requested_memory_mode": memory_mode,
                "tile_latent": tile_latent,
                "strategy": strategy,
                "memory": self.memory_snapshot(),
            },
        )

    async def generate(
        self,
        req: GenerationRequest,
        output_dir: Path,
        emit: Emit,
    ) -> dict[str, Any]:
        await self.load(req.model_id, req.quant, req.memory_mode, emit)
        req = replace(
            req,
            quant=self.loaded_quant or req.quant,
            memory_mode=self.loaded_memory_mode or req.memory_mode,
        )
        loop = asyncio.get_running_loop()
        return await self._run(self._generate_sync, req, output_dir, loop, emit)

    def memory_snapshot(self) -> dict[str, Any]:
        return _mlx_memory_snapshot()

    def _resolve_strategy(
        self,
        model_id: str,
        quant: str,
        memory_mode: str,
    ) -> tuple[dict[str, Any] | None, str, str]:
        spec = self._strategies.get(model_id)
        if spec is None:
            resolved_quant = "int4" if quant == "auto" else quant
            resolved_memory_mode = "resident" if memory_mode == "auto" else memory_mode
            return None, resolved_quant, resolved_memory_mode

        requested_precision = None if quant == "auto" else _precision(quant)
        requested_memory_mode = None if memory_mode == "auto" else _memory_mode(memory_mode)
        budget = _env_float("MXALLOY_MEMORY_BUDGET_GB")
        strategy = plan_execution(
            detect_device_profile(memory_budget_gb=budget),
            spec,
            requested_precision=requested_precision,
            requested_memory_mode=requested_memory_mode,
        )
        if not strategy.fits and (quant == "auto" or memory_mode == "auto"):
            raise RuntimeError(strategy.reason)
        return strategy.to_payload(), strategy.quant, strategy.memory_mode

    async def _run(self, fn: Callable[..., Any], *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(fn, *args))

    def _pipeline_class(self, model_id: str) -> Any:
        import importlib

        module_path, class_name = self._registry[model_id].split(":")
        return getattr(importlib.import_module(module_path), class_name)

    def _build(
        self, model_id: str, model_dir: str, quant_bits: int | None, tile_latent: int | None
    ) -> None:
        cls = self._pipeline_class(model_id)
        self._pipe = cls.from_pretrained(
            model_dir, quantize_bits=quant_bits, vae_tile_latent=tile_latent
        )

    def _dispose(self) -> None:
        self._pipe = None
        self._active_lora_key = ()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    def _apply_loras_sync(
        self,
        req: GenerationRequest,
        loop: asyncio.AbstractEventLoop,
        emit: Emit,
    ) -> dict[str, Any]:
        if self._pipe is None:
            raise RuntimeError("Pipeline is not loaded")
        active = tuple(
            (item.id, self._lora_resolver(item.id), float(item.strength))
            for item in req.loras
            if item.enabled
        )
        if active == self._active_lora_key:
            return {
                "active": [{"id": item[0], "strength": item[2]} for item in active],
                "applied": None,
                "skipped": [],
                "unchanged": True,
            }
        if not active:
            if self._active_lora_key:
                self._pipe.unload_lora_weights()
                _emit_sync(loop, emit, "lora", "LoRAs cleared", {"active": []})
            self._active_lora_key = ()
            return {"active": [], "applied": 0, "skipped": []}

        summary = self._pipe.set_lora_weights([(path, strength) for _, path, strength in active])
        applied = int(summary.get("applied", 0))
        if applied <= 0:
            raise RuntimeError("Selected LoRAs did not map to any modules in this model")
        payload = {
            **summary,
            "active": [{"id": item[0], "strength": item[2]} for item in active],
        }
        _emit_sync(loop, emit, "lora", f"LoRAs active ({applied} layers)", payload)
        self._active_lora_key = active
        return payload

    def _generate_sync(
        self,
        req: GenerationRequest,
        output_dir: Path,
        loop: asyncio.AbstractEventLoop,
        emit: Emit,
    ) -> dict[str, Any]:
        if self._pipe is None:
            raise RuntimeError("Pipeline is not loaded")

        seed = req.seed if req.seed is not None else _stable_seed(req.prompt)
        output_dir.mkdir(parents=True, exist_ok=True)
        _emit_sync(
            loop,
            emit,
            "generate",
            "Generation started",
            {
                "steps": req.steps,
                "width": req.width,
                "height": req.height,
                "seed": seed,
                "memory": self.memory_snapshot(),
            },
        )
        lora_summary = self._apply_loras_sync(req, loop, emit)

        def on_step(step: int, total: int) -> None:
            _emit_sync(
                loop,
                emit,
                "progress",
                f"Step {step}/{total}",
                {"step": step, "steps": total, "memory": self.memory_snapshot()},
            )

        result = self._pipe(
            req.prompt,
            seed=seed,
            num_inference_steps=req.steps,
            height=req.height,
            width=req.width,
            guidance=req.guidance,
            on_step=on_step,
        )
        image = result.images[0]

        filename = f"mxalloy_{req.model_id}_{int(time.time())}_{seed}.png"
        path = output_dir / filename
        image.save(path)
        meta = {
            "model_id": req.model_id,
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": req.width,
            "height": req.height,
            "steps": req.steps,
            "guidance": req.guidance,
            "seed": seed,
            "quant": req.quant,
            "memory_mode": req.memory_mode,
            "refs": req.refs,
            "loras": [asdict(item) for item in req.loras],
            "lora_summary": lora_summary,
            "mock": False,
            "created_at": time.time(),
        }
        path.with_suffix(".json").write_text(_json_dumps(meta), encoding="utf-8")
        _emit_sync(
            loop,
            emit,
            "complete",
            "Image written",
            {"path": str(path), "seed": seed, "memory": self.memory_snapshot()},
        )
        return {"path": path, "meta": meta}


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _render_mock_image(req: GenerationRequest, path: Path, seed: int) -> None:
    width = max(256, min(int(req.width), 2048))
    height = max(256, min(int(req.height), 2048))
    bg = "#f6f6f2"
    ink = "#111111"
    line = "#d0d0ca"
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    step = max(64, min(width, height) // 8)
    for x in range(0, width, step):
        draw.line((x, 0, x, height), fill=line, width=1)
    for y in range(0, height, step):
        draw.line((0, y, width, y), fill=line, width=1)

    margin = max(24, min(width, height) // 18)
    draw.rectangle((margin, margin, width - margin, height - margin), outline=ink, width=2)
    draw.line((margin, margin, width - margin, height - margin), fill=ink, width=1)
    draw.line((width - margin, margin, margin, height - margin), fill=ink, width=1)

    lines = [
        "mxalloy local tester",
        f"{req.model_id} / {req.quant} / {req.memory_mode}",
        f"{width}x{height}  steps {req.steps}  seed {seed}",
    ]
    if req.loras:
        active = [item.id for item in req.loras if item.enabled]
        lines.append(f"LoRA: {', '.join(active[:3])}")
    if req.refs:
        lines.append(f"Refs: {len(req.refs)}")
    lines.extend(textwrap.wrap(req.prompt or "No prompt", width=52)[:6])

    text_x = margin + 18
    text_y = margin + 18
    for line_text in lines:
        draw.text((text_x, text_y), line_text, fill=ink, font=font)
        text_y += 16

    image.save(path)


def _json_dumps(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _emit_sync(
    loop: asyncio.AbstractEventLoop,
    emit: Emit,
    kind: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> None:
    future = asyncio.run_coroutine_threadsafe(emit(kind, message, payload), loop)
    future.result(timeout=10)


def _quant_bits(quant: str) -> int | None:
    normalized = quant.lower()
    if normalized == "int4":
        return 4
    if normalized == "int8":
        return 8
    if normalized in {"bf16", "fp16", "none"}:
        return None
    raise ValueError(f"Unsupported quant: {quant}")


def _precision(quant: str):
    normalized = quant.lower()
    if normalized in {"bf16", "int8", "int4"}:
        return normalized
    if normalized in {"fp16", "none"}:
        return "bf16"
    raise ValueError(f"Unsupported quant: {quant}")


def _memory_mode(memory_mode: str):
    normalized = memory_mode.lower()
    if normalized in {"resident", "staged", "survival"}:
        return normalized
    raise ValueError(f"Unsupported memory mode: {memory_mode}")


def _tile_latent(memory_mode: str) -> int | None:
    return {
        "resident": 128,
        "staged": 96,
        "survival": 64,
    }.get(memory_mode, 128)


def _mlx_memory_snapshot() -> dict[str, Any]:
    try:
        import mlx.core as mx

        active = int(mx.get_active_memory()) if hasattr(mx, "get_active_memory") else 0
        peak = int(mx.get_peak_memory()) if hasattr(mx, "get_peak_memory") else active
        cache = int(mx.get_cache_memory()) if hasattr(mx, "get_cache_memory") else 0
    except Exception as exc:
        return {
            "available": False,
            "source": "mlx",
            "error": str(exc),
            "active_gb": 0.0,
            "peak_gb": 0.0,
            "cache_gb": 0.0,
            "label": "unavailable",
        }
    return _memory_payload(
        active_gb=_bytes_to_gb(active),
        peak_gb=_bytes_to_gb(peak),
        cache_gb=_bytes_to_gb(cache),
        available=True,
        source="mlx",
    )


def _memory_payload(
    *,
    active_gb: float,
    peak_gb: float,
    cache_gb: float,
    available: bool,
    source: str,
) -> dict[str, Any]:
    total = _system_memory_gb()
    percent = round(min(100.0, active_gb / total * 100), 1) if total else None
    return {
        "available": available,
        "source": source,
        "active_gb": round(active_gb, 2),
        "peak_gb": round(peak_gb, 2),
        "cache_gb": round(cache_gb, 2),
        "system_total_gb": total,
        "percent": percent,
        "label": f"{active_gb:.2f} GB",
    }


def _bytes_to_gb(value: int) -> float:
    return value / 1024**3


def _system_memory_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round((pages * page_size) / 1024**3, 1)
    except (AttributeError, OSError, ValueError):
        return None


def _env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
