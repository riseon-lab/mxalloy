"""Engine adapter boundary for the local tester surface.

The UI can be built and exercised before the real generation graph is stable. Later,
replace ``MockEngine`` with an adapter around the resident mxalloy engine while keeping
the server/frontend contract intact.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import textwrap
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

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


class RealFlux2KleinEngine:
    """Adapter around the resident native FLUX.2-klein engine used by the tester UI."""

    def __init__(self, model_dir_resolver: Callable[[str], str]) -> None:
        self.mode = "real"
        self.loaded_model_id: str | None = None
        self.loaded_quant: str | None = None
        self.loaded_memory_mode: str | None = None
        self._engine: Any | None = None
        self._model_dir_resolver = model_dir_resolver
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mxalloy-surface")

    async def load(self, model_id: str, quant: str, memory_mode: str, emit: Emit) -> None:
        if (
            self._engine is not None
            and self.loaded_model_id == model_id
            and self.loaded_quant == quant
            and self.loaded_memory_mode == memory_mode
        ):
            await emit(
                "ready",
                "Model already warm",
                {"model_id": model_id, "memory": self.memory_snapshot()},
            )
            return

        quant_bits = _quant_bits(quant)
        tile_latent = _tile_latent(memory_mode)
        model_dir = self._model_dir_resolver(model_id)
        await emit(
            "load",
            f"Loading {model_id}",
            {
                "model_id": model_id,
                "quant": quant,
                "memory_mode": memory_mode,
                "model_dir": model_dir,
                "memory": self.memory_snapshot(),
            },
        )
        await self._run(self._dispose)
        await self._run(self._build, model_dir, quant_bits, tile_latent)
        self.loaded_model_id = model_id
        self.loaded_quant = quant
        self.loaded_memory_mode = memory_mode
        await emit(
            "ready",
            "Real engine ready",
            {
                "model_id": model_id,
                "quant": quant,
                "memory_mode": memory_mode,
                "tile_latent": tile_latent,
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
        loop = asyncio.get_running_loop()
        return await self._run(self._generate_sync, req, output_dir, loop, emit)

    def memory_snapshot(self) -> dict[str, Any]:
        return _mlx_memory_snapshot()

    async def _run(self, fn: Callable[..., Any], *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, partial(fn, *args))

    def _build(self, model_dir: str, quant_bits: int | None, tile_latent: int | None) -> None:
        from mxalloy.models.flux2.engine import Flux2KleinEngine

        self._engine = Flux2KleinEngine(
            model_dir=model_dir,
            quantize_bits=quant_bits,
            vae_tile_latent=tile_latent,
        )

    def _dispose(self) -> None:
        self._engine = None
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    def _generate_sync(
        self,
        req: GenerationRequest,
        output_dir: Path,
        loop: asyncio.AbstractEventLoop,
        emit: Emit,
    ) -> dict[str, Any]:
        if self._engine is None:
            raise RuntimeError("Real engine is not loaded")

        import mlx.core as mx

        from mxalloy.models.flux2.engine import _TEXT_ENCODER_OUT_LAYERS
        from mxalloy.models.flux2.latents import prepare_packed_latents, prepare_text_ids
        from mxalloy.models.flux2.scheduler import FlowMatchEulerScheduler

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

        input_ids, attention_mask = self._engine.tokenizer.encode(req.prompt)
        prompt_embeds = self._engine.text_encoder.get_prompt_embeds(
            input_ids, attention_mask, _TEXT_ENCODER_OUT_LAYERS
        )
        text_ids = prepare_text_ids(prompt_embeds)
        mx.eval(prompt_embeds)
        _emit_sync(
            loop,
            emit,
            "progress",
            "Prompt encoded",
            {"step": 0, "steps": req.steps, "memory": self.memory_snapshot()},
        )

        latents, latent_ids, latent_height, latent_width = prepare_packed_latents(
            seed=seed, height=req.height, width=req.width, batch_size=1
        )
        image_seq_len = (req.height // 16) * (req.width // 16)
        scheduler = FlowMatchEulerScheduler(
            num_inference_steps=req.steps, image_seq_len=image_seq_len
        )

        for idx in range(req.steps):
            noise = self._engine.transformer(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=scheduler.timesteps[idx],
                img_ids=latent_ids,
                txt_ids=text_ids,
                guidance=req.guidance,
            )
            latents = scheduler.step(noise, idx, latents)
            mx.eval(latents)
            _emit_sync(
                loop,
                emit,
                "progress",
                f"Step {idx + 1}/{req.steps}",
                {
                    "step": idx + 1,
                    "steps": req.steps,
                    "memory": self.memory_snapshot(),
                },
            )

        packed = latents.reshape(1, latent_height, latent_width, latents.shape[-1]).transpose(
            0, 3, 1, 2
        )
        decoded = self._engine.vae.decode_packed_latents(
            packed, tile_latent=self._engine.vae_tile_latent
        )
        mx.eval(decoded)
        image = self._engine._to_pil(decoded)

        filename = f"mxalloy_flux2_klein_{int(time.time())}_{seed}.png"
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
