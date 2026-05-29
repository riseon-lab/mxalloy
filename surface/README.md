# mxalloy Local Tester Surface

Small local UI for dogfooding mxalloy while the engine is still moving.

Run it from the repo root:

```bash
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

The current engine path is a mock adapter. It exercises the UI, settings, LoRA upload,
reference images, output gallery, progress events, and API contract without waiting for
the real resident generation graph.

The real engine should replace `MockEngine` behind the same boundary in
`surface/engine.py`.
