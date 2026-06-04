"""FastAPI server for the local mxalloy tester surface."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import stat
import time
from contextlib import suppress
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from surface.engine import GenerationRequest, LoraSelection, MockEngine, RealPipelineEngine

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SURFACE_HOME = Path(os.environ.get("MXALLOY_SURFACE_HOME", Path.home() / ".config" / "mxalloy"))
SETTINGS_PATH = SURFACE_HOME / "surface.json"
SECRETS_PATH = SURFACE_HOME / "secrets.json"
DOWNLOADS_DIR = SURFACE_HOME / "downloads"

DEFAULT_SETTINGS = {
    "model_cache_path": os.environ.get(
        "MXALLOY_MODEL_CACHE",
        str(Path.home() / ".cache" / "huggingface"),
    ),
    "lora_folder_path": os.environ.get(
        "MXALLOY_LORA_DIR",
        str(Path.home() / "mxalloy" / "loras"),
    ),
    "output_folder_path": os.environ.get(
        "MXALLOY_OUTPUT_DIR",
        str(Path.home() / "mxalloy" / "outputs"),
    ),
    "reference_folder_path": os.environ.get(
        "MXALLOY_REF_DIR",
        str(Path.home() / "mxalloy" / "refs"),
    ),
}

MODEL_REGISTRY = [
    {
        "id": "flux2-klein-4b",
        "name": "FLUX.2 klein 4B",
        "status": "target",
        "description": "Available local test target. Native MLX generation with tiled VAE.",
        "default_width": 1024,
        "default_height": 1024,
        "default_steps": 4,
        "default_guidance": 1.0,
        "quants": ["int4", "int8", "bf16"],
        "memory_modes": ["resident", "staged", "survival"],
        "license": "Apache-2.0",
        "notes": {
            "int4": "lowest memory",
            "int8": "quality default",
            "bf16": "unquantized baseline",
            "resident": "warm model, 1024px VAE tile",
            "staged": "smaller VAE tile",
            "survival": "smallest VAE tile",
        },
    },
    {
        "id": "z-image-turbo",
        "name": "Z-Image Turbo 6B",
        "status": "target",
        "description": "Alibaba Tongyi S3-DiT (clean-room MLX). 8-step, guidance-free.",
        "default_width": 1024,
        "default_height": 1024,
        "default_steps": 8,
        "default_guidance": 0.0,
        "quants": ["int4", "int8", "bf16"],
        "memory_modes": ["resident"],
        "license": "Apache-2.0",
        "notes": {
            "int4": "lowest memory",
            "int8": "higher quality",
            "bf16": "unquantized baseline",
            "resident": "warm model, full VAE decode",
        },
    },
]


# model_id -> "module:ClassName" of its MXPipeline. Add a line here to surface a new model.
PIPELINES = {
    "flux2-klein-4b": "mxdiffusers.flux.pipeline:MXFluxPipeline",
    "z-image-turbo": "mxdiffusers.zimage.pipeline:MXZimagePipeline",
}

# model_id -> Hugging Face cache repo dir (for resolving the local snapshot).
_HF_REPOS = {
    "flux2-klein-4b": "models--black-forest-labs--FLUX.2-klein-4B",
    "z-image-turbo": "models--Tongyi-MAI--Z-Image-Turbo",
}


def _create_engine() -> MockEngine | RealPipelineEngine:
    if os.environ.get("MXALLOY_SURFACE_ENGINE", "real").lower() == "mock":
        return MockEngine()
    return RealPipelineEngine(lambda model_id: _model_dir_for(model_id), PIPELINES)


app = FastAPI(title="mxalloy local tester")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

engine = _create_engine()
logs: list[dict[str, Any]] = []
subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
current_task: asyncio.Task | None = None
current_job_id: str | None = None


class LoraSelectionBody(BaseModel):
    id: str
    strength: float = Field(1.0, ge=0.0, le=2.0)
    enabled: bool = True


class GenerateBody(BaseModel):
    model_id: str = "flux2-klein-4b"
    prompt: str = ""
    negative_prompt: str = ""
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    steps: int = Field(4, ge=1, le=100)
    guidance: float = Field(1.0, ge=0.0, le=20.0)
    seed: int | None = None
    quant: str = "int4"
    memory_mode: str = "resident"
    refs: list[str] = Field(default_factory=list)
    loras: list[LoraSelectionBody] = Field(default_factory=list)


class LoadBody(BaseModel):
    model_id: str = "flux2-klein-4b"
    quant: str = "int4"
    memory_mode: str = "resident"


class SettingsBody(BaseModel):
    hf_token: str | None = None
    clear_hf_token: bool = False
    model_cache_path: str | None = None
    lora_folder_path: str | None = None
    output_folder_path: str | None = None
    reference_folder_path: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> dict[str, Any]:
    task_running = current_task is not None and not current_task.done()
    return {
        "engine": {
            "mode": getattr(engine, "mode", "unknown"),
            "loaded_model_id": engine.loaded_model_id,
            "quant": engine.loaded_quant,
            "memory_mode": engine.loaded_memory_mode,
            "running": task_running,
            "job_id": current_job_id if task_running else None,
            "memory": engine.memory_snapshot(),
        },
        "settings": public_settings(),
        "logs": logs[-80:],
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    items = []
    for model in MODEL_REGISTRY:
        row = dict(model)
        local_path = _local_model_path(row["id"])
        row["downloaded"] = local_path is not None
        row["available"] = local_path is not None
        row["local_path"] = str(local_path) if local_path else None
        items.append(row)
    return {"models": items}


@app.post("/api/models/download")
async def download_model(body: LoadBody) -> dict[str, Any]:
    model = _model_or_404(body.model_id)
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    await publish("download", f"Marked {model['name']} as available", {"model_id": body.model_id})
    _download_marker(body.model_id).write_text(json.dumps({"ts": time.time()}), encoding="utf-8")
    return {"status": "downloaded", "model_id": body.model_id}


@app.post("/api/load")
async def load(body: LoadBody) -> dict[str, Any]:
    _validate_model_options(body.model_id, body.quant, body.memory_mode)
    pulse = asyncio.create_task(_memory_pulse("load"))
    try:
        await engine.load(body.model_id, body.quant, body.memory_mode, publish)
    finally:
        pulse.cancel()
        with suppress(asyncio.CancelledError):
            await pulse
    return {"status": "ready", "model_id": body.model_id}


@app.post("/api/generate")
async def generate(body: GenerateBody) -> dict[str, Any]:
    global current_task, current_job_id
    if current_task is not None and not current_task.done():
        raise HTTPException(409, "Generation already running")
    if not body.prompt.strip():
        raise HTTPException(400, "Prompt is required")
    _validate_model_options(body.model_id, body.quant, body.memory_mode)

    job_id = secrets.token_hex(8)
    current_job_id = job_id
    req = GenerationRequest(
        model_id=body.model_id,
        prompt=body.prompt,
        negative_prompt=body.negative_prompt,
        width=body.width,
        height=body.height,
        steps=body.steps,
        guidance=body.guidance,
        seed=body.seed,
        quant=body.quant,
        memory_mode=body.memory_mode,
        refs=body.refs,
        loras=[
            LoraSelection(id=item.id, strength=item.strength, enabled=item.enabled)
            for item in body.loras
        ],
    )
    current_task = asyncio.create_task(_run_generation(job_id, req))
    return {"status": "started", "job_id": job_id}


@app.post("/api/cancel")
async def cancel() -> dict[str, Any]:
    if current_task is None or current_task.done():
        return {"status": "idle"}
    current_task.cancel()
    await publish("cancel", "Cancellation requested", {"job_id": current_job_id})
    return {"status": "cancelling", "job_id": current_job_id}


@app.get("/api/events")
async def events() -> StreamingResponse:
    async def stream():
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        subscribers.add(queue)
        try:
            for event in logs[-20:]:
                yield _sse(event)
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield _sse(event)
                except TimeoutError:
                    yield ": ping\n\n"
        finally:
            subscribers.discard(queue)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/loras")
def list_loras() -> dict[str, Any]:
    folder = _lora_dir()
    folder.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".safetensors", ".ckpt", ".pt", ".bin"}:
            rel = path.relative_to(folder).as_posix()
            items.append(
                {
                    "id": rel,
                    "name": path.name,
                    "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
                    "updated_at": path.stat().st_mtime,
                }
            )
    return {"loras": items}


@app.post("/api/loras")
async def upload_lora(file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
    filename = _safe_filename(file.filename or "adapter.safetensors")
    if Path(filename).suffix.lower() not in {".safetensors", ".ckpt", ".pt", ".bin"}:
        raise HTTPException(400, "Unsupported LoRA file type")
    dest = _lora_dir() / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    await publish("lora", "LoRA added", {"id": filename})
    return {"status": "ok", "lora": {"id": filename, "name": filename}}


@app.delete("/api/loras/{lora_id:path}")
async def delete_lora(lora_id: str) -> dict[str, Any]:
    path = (_lora_dir() / lora_id).resolve()
    if _lora_dir().resolve() not in path.parents:
        raise HTTPException(400, "Invalid LoRA path")
    path.unlink(missing_ok=True)
    await publish("lora", "LoRA removed", {"id": lora_id})
    return {"status": "deleted", "id": lora_id}


@app.get("/api/assets")
def list_assets() -> dict[str, Any]:
    return {"assets": _collect_assets()}


@app.post("/api/assets/upload")
async def upload_asset(file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
    filename = _safe_filename(file.filename or "reference.png")
    if Path(filename).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(400, "Unsupported reference image type")
    dest = _refs_dir() / f"{int(time.time())}_{filename}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    await publish("asset", "Reference image added", {"id": f"refs/{dest.name}"})
    return {"status": "ok", "asset": _asset_for_path(dest, "reference")}


@app.get("/api/assets/file/{scope}/{filename:path}")
def asset_file(scope: str, filename: str) -> FileResponse:
    roots = {"outputs": _output_dir(), "refs": _refs_dir()}
    if scope not in roots:
        raise HTTPException(404, "Unknown asset scope")
    path = (roots[scope] / filename).resolve()
    if roots[scope].resolve() not in path.parents:
        raise HTTPException(400, "Invalid asset path")
    if not path.exists() or path.suffix.lower() == ".json":
        raise HTTPException(404, "Asset not found")
    return FileResponse(path)


@app.delete("/api/assets/{scope}/{filename:path}")
async def delete_asset(scope: str, filename: str) -> dict[str, Any]:
    roots = {"outputs": _output_dir(), "refs": _refs_dir()}
    if scope not in roots:
        raise HTTPException(404, "Unknown asset scope")
    path = (roots[scope] / filename).resolve()
    if roots[scope].resolve() not in path.parents:
        raise HTTPException(400, "Invalid asset path")
    path.unlink(missing_ok=True)
    meta = path.with_suffix(".json")
    meta.unlink(missing_ok=True)
    await publish("asset", "Asset deleted", {"scope": scope, "filename": filename})
    return {"status": "deleted"}


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    return public_settings()


@app.post("/api/settings")
def update_settings(body: SettingsBody) -> dict[str, Any]:
    settings = _read_settings()
    for key in (
        "model_cache_path",
        "lora_folder_path",
        "output_folder_path",
        "reference_folder_path",
    ):
        value = getattr(body, key)
        if value:
            settings[key] = str(Path(value).expanduser())
    _write_private_json(SETTINGS_PATH, settings)

    if body.clear_hf_token:
        _write_private_json(SECRETS_PATH, {})
    elif body.hf_token is not None and body.hf_token.strip():
        _write_private_json(SECRETS_PATH, {"hf_token": body.hf_token.strip()})

    for path in (_lora_dir(), _output_dir(), _refs_dir()):
        path.mkdir(parents=True, exist_ok=True)
    return public_settings()


@app.post("/api/settings/hf-token/test")
def test_hf_token() -> dict[str, Any]:
    token = _read_secrets().get("hf_token", "")
    if not token:
        return {"status": "missing", "message": "No token saved"}
    if token.startswith("hf_") and len(token) > 12:
        return {"status": "format_ok", "message": "Token format looks valid"}
    return {
        "status": "format_warning",
        "message": "Token saved, but it does not look like an HF token",
    }


async def _run_generation(job_id: str, req: GenerationRequest) -> None:
    global current_job_id
    pulse = asyncio.create_task(_memory_pulse(job_id))
    try:
        result = await engine.generate(req, _output_dir(), publish)
        await publish(
            "asset",
            "Output available",
            {"job_id": job_id, "asset": _asset_for_path(result["path"], "output")},
        )
    except asyncio.CancelledError:
        await publish("cancelled", "Generation cancelled", {"job_id": job_id})
    except Exception as exc:
        await publish("error", f"{type(exc).__name__}: {exc}", {"job_id": job_id})
    finally:
        pulse.cancel()
        with suppress(asyncio.CancelledError):
            await pulse
        current_job_id = None


async def publish(kind: str, message: str, payload: dict[str, Any] | None = None) -> None:
    event = {"ts": time.time(), "kind": kind, "message": message, "payload": payload or {}}
    logs.append(event)
    del logs[:-200]
    for queue in list(subscribers):
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            pass


def _sse(event: dict[str, Any]) -> str:
    return f"event: {event['kind']}\ndata: {json.dumps(event)}\n\n"


@app.get("/api/memory")
def memory() -> dict[str, Any]:
    return {"memory": engine.memory_snapshot()}


async def _memory_pulse(job_id: str) -> None:
    try:
        while True:
            await publish(
                "memory",
                "Memory sample",
                {"job_id": job_id, "memory": engine.memory_snapshot()},
            )
            await asyncio.sleep(0.75)
    except asyncio.CancelledError:
        await publish(
            "memory",
            "Memory sample",
            {"job_id": job_id, "memory": engine.memory_snapshot()},
        )
        raise


def _model_or_404(model_id: str) -> dict[str, Any]:
    for model in MODEL_REGISTRY:
        if model["id"] == model_id:
            return model
    raise HTTPException(404, f"Unknown model: {model_id}")


def _validate_model_options(model_id: str, quant: str, memory_mode: str) -> dict[str, Any]:
    model = _model_or_404(model_id)
    if quant not in model["quants"]:
        raise HTTPException(400, f"{model['name']} does not support quant {quant!r}")
    if memory_mode not in model["memory_modes"]:
        raise HTTPException(400, f"{model['name']} does not support memory mode {memory_mode!r}")
    return model


def _model_dir_for(model_id: str) -> str:
    _model_or_404(model_id)
    path = _local_model_path(model_id)
    if path is None:
        raise FileNotFoundError(
            f"{model_id} was not found in the configured model cache. "
            "Check Settings > Model cache, or download the model first."
        )
    return str(path)


def _local_model_path(model_id: str) -> Path | None:
    repo = _HF_REPOS.get(model_id)
    if repo is None:
        return None
    root = Path(_read_settings()["model_cache_path"]).expanduser()
    candidates = []
    if _looks_like_model_snapshot(root):
        candidates.append(root)
    for base in (root / "hub" / repo, root / repo):
        candidates.extend(sorted((base / "snapshots").glob("*")))
    candidates.extend(sorted(root.glob("snapshots/*")))
    valid = [path for path in candidates if _looks_like_model_snapshot(path)]
    return valid[-1] if valid else None


def _looks_like_model_snapshot(path: Path) -> bool:
    return (
        (path / "transformer").is_dir()
        and (path / "text_encoder").is_dir()
        and (path / "vae").is_dir()
        and (path / "tokenizer").is_dir()
    )


def _read_settings() -> dict[str, str]:
    if SETTINGS_PATH.exists():
        try:
            loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            return {**DEFAULT_SETTINGS, **{k: str(v) for k, v in loaded.items() if v}}
        except Exception:
            return dict(DEFAULT_SETTINGS)
    return dict(DEFAULT_SETTINGS)


def _read_secrets() -> dict[str, str]:
    if not SECRETS_PATH.exists():
        return {}
    try:
        return json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def public_settings() -> dict[str, Any]:
    settings = _read_settings()
    return {
        **settings,
        "hf_token_set": bool(_read_secrets().get("hf_token")),
        "settings_path": str(SETTINGS_PATH),
    }


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def _lora_dir() -> Path:
    return Path(_read_settings()["lora_folder_path"]).expanduser()


def _output_dir() -> Path:
    return Path(_read_settings()["output_folder_path"]).expanduser()


def _refs_dir() -> Path:
    return Path(_read_settings()["reference_folder_path"]).expanduser()


def _download_marker(model_id: str) -> Path:
    return DOWNLOADS_DIR / f"{_safe_filename(model_id)}.json"


def _collect_assets() -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for path in _sorted_by_mtime(_output_dir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            assets.append(_asset_for_path(path, "output"))
    for path in _sorted_by_mtime(_refs_dir()):
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            assets.append(_asset_for_path(path, "reference"))
    return assets


def _sorted_by_mtime(root: Path) -> list[Path]:
    return sorted(
        root.glob("*"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )


def _asset_for_path(path: Path, role: str) -> dict[str, Any]:
    scope = "outputs" if role == "output" else "refs"
    root = _output_dir() if role == "output" else _refs_dir()
    rel = path.relative_to(root).as_posix()
    meta_path = path.with_suffix(".json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return {
        "id": f"{scope}/{rel}",
        "role": role,
        "name": path.name,
        "url": f"/api/assets/file/{scope}/{rel}",
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "updated_at": path.stat().st_mtime,
        "meta": meta,
    }


def _safe_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("._")
    return cleaned or "file"
