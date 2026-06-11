# mxalloy Local Tester Surface

Small local UI for dogfooding mxalloy while the engine is still moving. It lives only in the
repo checkout — it is not part of the published wheel.

Run it from the repo root:

```bash
pip install -e ".[mlx,surface]"   # the [surface] extra provides fastapi/uvicorn
python3 -m uvicorn surface.server:app --reload --port 8787
```

Open `http://127.0.0.1:8787`.

Useful local overrides:

```bash
MXALLOY_SURFACE_HOME=/path/to/config \
MXALLOY_OUTPUT_DIR=/path/to/outputs \
MXALLOY_LORA_DIR=/path/to/loras \
MXALLOY_REF_DIR=/path/to/refs \
python3 -m uvicorn surface.server:app --reload --port 8787
```

By default the surface uses the real resident pipeline engine. Set
`MXALLOY_SURFACE_ENGINE=mock` when you only want to exercise the UI, settings, LoRA upload,
reference images, output gallery, progress events, and API contract without loading models.
