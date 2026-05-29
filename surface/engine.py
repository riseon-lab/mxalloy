"""Engine adapter boundary for the local tester surface.

The UI can be built and exercised before the real generation graph is stable. Later,
replace ``MockEngine`` with an adapter around the resident mxalloy engine while keeping
the server/frontend contract intact.
"""

from __future__ import annotations

import asyncio
import hashlib
import textwrap
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
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
        self.loaded_model_id: str | None = None
        self.loaded_quant: str | None = None
        self.loaded_memory_mode: str | None = None

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
            ("reading manifest", "0.4 GB"),
            ("streaming quantized weights", "3.1 GB"),
            ("warming graph", "4.6 GB"),
        ):
            await asyncio.sleep(0.25)
            await emit("load", label, {"active_memory": memory})

        self.loaded_model_id = model_id
        self.loaded_quant = quant
        self.loaded_memory_mode = memory_mode
        await emit(
            "ready",
            "Mock engine ready",
            {"model_id": model_id, "quant": quant, "memory_mode": memory_mode},
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
            await emit(
                "progress",
                f"Step {step}/{req.steps}",
                {"step": step, "steps": req.steps, "active_memory": "4.9 GB"},
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
