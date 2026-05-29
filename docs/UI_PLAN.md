# Local UI Plan

## Decision

The surface is the **dogfood harness for the engine**, not a product or a distribution moat —
the moat is the optimization layer (fused primitives + streaming loader + tiled VAE). Keep it a
thin tester over the resident engine manager.

Build a lean mxalloy-native local tester before any IgglePixel fork. IgglePixel remains useful reference material for registry-driven controls, runner lifecycle, LoRA conventions, assets, and preview mode, but the first mxalloy surface should be small enough to change quickly while the engine is still moving.

## Product Shape

- Single local generation workspace as the first screen.
- Static HTML/vanilla JS or a tiny app shell.
- Small FastAPI backend with one mxalloy engine manager.
- Black/white/solid colour system, dense controls, no glass, no drop shadows, no landing page.
- Local-only, single-user, dogfooding/tester posture.

## Views

- **Generate**: prompt, negative prompt, refs, controls, progress, logs, memory status, and output viewer.
- **Models**: curated Mac-fitting models, download/load status, quant and memory mode.
- **LoRAs**: add local LoRA files, remove them, set strength, see compatibility/status.
- **Outputs**: generated image gallery, metadata, reveal/download/delete.
- **Settings**: HF token, model cache path, LoRA folder, output folder.

## Generate Controls

- Prompt and negative prompt.
- Reference image upload/selection.
- Model and variant.
- Quant mode.
- Memory mode: `resident`, `staged`, `survival`.
- Width, height, steps, seed, guidance/CFG where supported.
- LoRA list with enable toggles and strength sliders.
- Generate and cancel.
- Progress, logs, memory readout, loaded component status.

## Secrets And Paths

- HF token is needed for gated/private model downloads.
- Do not store tokens in browser localStorage.
- Backend storage order:
  1. macOS Keychain if practical.
  2. `~/.config/mxalloy/secrets.json` with `0600` permissions.
  3. Ephemeral session-only token.
- Settings should expose:
  - HF token: masked input, save, clear, test token.
  - Model cache path.
  - LoRA folder path.
  - Output folder path.

## Backend API Sketch

- `GET /api/status`
- `GET /api/models`
- `POST /api/models/download`
- `POST /api/load`
- `POST /api/generate`
- `POST /api/cancel`
- `GET /api/events` for SSE progress/logs.
- `GET /api/loras`
- `POST /api/loras`
- `DELETE /api/loras/{id}`
- `GET /api/assets`
- `POST /api/assets/upload`
- `GET /api/settings`
- `POST /api/settings`
- `POST /api/settings/hf-token/test`

## Deferred

- CivitAI browsing/downloads.
- Auth gate.
- At-rest encryption.
- Moderation.
- Trainers.
- RunPod launcher.
- Multi-media catalogue.
- Full IgglePixel fork.
