# mxalloy v0.1 — Distribution-Readiness Report

Produced by a six-dimension audit (packaging, public API, docs, dead-code/fiction, tests/CI,
licensing) + release-manager synthesis, then **verified by hand** on the load-bearing claims.
This is the punch-list between "works on my machine + beats mflux" and "a stranger can
`pip install` and generate an image."

## Verdict: NO-GO for immediate release — but fixable without a code rewrite

The package internals are sound (pure-Python, correctly-scoped wheel, `import mxalloy` proven
mlx-free, core↛models boundary holds, mflux not a runtime import, no secrets/weights committed,
the streaming-load/tiled-VAE/LoRA engine is real and beats the reference). The gaps are
**attribution, packaging hygiene, doc honesty, and automation** — not the code.

## 2026-06-10 Review Update — blockers cleared

All nine blockers are now resolved; the remaining gate lives in `docs/RELEASE_CHECKLIST.md`.

- **B7 fixed**: `component_files`/`load_quantized` raise `ModelLoadError`; the planner and
  `detect_device_profile` raise `ConfigurationError`; never-raised types (Quantization/
  IncompatibleLoRA/UnsupportedHardware) deleted; `docs/ERRORS.md` rewritten to match.
- **B8 fixed**: dead `AlloyConfig`/`QuantizationConfig`/`RuntimeConfig` deleted along with
  `kernels/`, `runtime/scheduler.py`, `utils/`; VERSIONING blesses the real surface
  (loader + errors + runtime planning).
- **B9 fixed**: `.github/workflows/ci.yml` — lint, ubuntu-no-mlx pytest (3.11/3.12),
  wheel-build + twine + clean-env install smoke, macOS+mlx pytest + strict mypy.
- Should-fixes closed: phantom deps removed (diffusers, hf-hub, packaging, safetensors,
  tqdm), classifiers/keywords added, `mlx>=0.31` floor, `py.typed` shipped (core is
  mypy-strict clean), README rewritten (logo absolute URL, attention/tiling claims
  corrected, quickstart works via the new HF-cache resolver in `mxdiffusers/hub.py`).
- Open decision 3 (author identity) remains the maintainer's; see RELEASE_CHECKLIST.

## 2026-06-05 Review Update

- Root `NOTICE` now carries the mflux MIT notice for the FLUX port lineage.
- `pyproject.toml` repository URLs are no longer placeholders, and the wheel package list now
  includes `mxdiffusers` and `mxtts`.
- README/provenance now avoid calling FLUX clean-room. Z-Image is described as a clean-room
  transformer with shared FLUX-derived Qwen/VAE helpers until those are independently split.
- `.gitignore` now excludes local generated smoke-output folders.

## Hand-verified corrections to the audit

- **FLUX.2-klein-4B license — CONFIRMED Apache-2.0, commercial use permitted** (HF model card:
  "Open weights available for commercial use under the Apache 2.0 license"). The audit's
  "probably non-commercial" was a **false alarm**; our docs are correct. *Not a blocker.*
- **B1 severity — attribution gap, not verbatim theft.** mflux has no `_empirical_mu` /
  `_apply_rope_bshd` symbols, so "byte-identical verbatim copy" is unconfirmed. The accurate
  framing: the `flux2` modules were built as a **close port of / verified against mflux** (our
  own docstrings: "faithful port… mirror the reference exactly"; commit history: "port X,
  verified vs mflux"). That owes MIT attribution and the docstrings need rewording. The fix is
  identical regardless of how literal the copying was.

## Blockers (must-fix before any public release)

| # | Issue | Where | Fix |
|---|-------|-------|-----|
| **B1** | mflux MIT attribution must remain attached to the FLUX close port | `mxdiffusers/flux/*`, `NOTICE` | Root `NOTICE` added. Keep per-file/provenance wording honest until FLUX helpers are independently re-derived. |
| **B2** | sdist must avoid leaking local/generated artifacts | `pyproject.toml`, `.gitignore` | sdist include list and `.claude/`/`experiments/_*/` ignores added; rebuild + inspect both wheel and sdist before release. |
| **B3** | Placeholder URLs must not ship in metadata | `pyproject.toml` | Fixed to `github.com/riseon-lab/mxalloy`; add Docs URL when docs are public. |
| **B4** | Docs must not sell the compiled Metal kernel as shipped | `README.md`, `docs/DESIGN_BRIEF.md`, `docs/SCHEDULE.md` | README and planning docs now frame the Metal primitive as research/frozen until gates pass. Recheck before release. |
| **B5** | No documented path to generate an image; the vision's `mxalloy.loader(...)` **collides with the `loader` module** (not callable) | `README.md`, `docs/VERSIONING.md:13`, `engine.py:36` | Bless `Flux2KleinEngine` as the v0.1 public entry; write a copy-paste Quickstart. Rename the aspirational front door to `mxalloy.load(...)`/`generate(...)`, mark not-yet-shipped. *(see Decision 2)* |
| **B6** | `pip install mxalloy` can't run anything (mlx is an extra) | `README.md:9,20` | Use `pip install "mxalloy[mlx]"` everywhere a user installs *to run*. |
| **B7** | Documented `mxalloy.errors` hierarchy is **never raised** (loader raises `FileNotFoundError`) | `errors.py`, `loader.py:40` | Wire `loader.py:40 → ModelLoadError` (subclasses `RuntimeError`, stays catchable); drop error types v0.1 won't raise. |
| **B8** | Two quant configs: public `AlloyConfig/QuantizationConfig` are **dead**; live one is `loader.QuantConfig` | `config.py`, `__init__.py`, `loader.py` | Drop the dead dataclasses from `__all__`+VERSIONING; document `QuantConfig` as the single public quant config. |
| **B9** | No CI — every distribution invariant rests on one local Mac | repo root | `.github/workflows/ci.yml`: lint+mypy, ubuntu clean-install + import-mlx-free smoke (against the *installed wheel*), ubuntu pytest, macOS-arm64 `[dev,mlx]` pytest. |

## Should-fix (before or shortly after release)

- **Peak-memory regression test** — the whole thesis (4-bit load ~4.5 GB) is asserted nowhere; add an mlx-gated test asserting peak `< 0.6×` full-bf16. *Converts the headline into a merge gate.*
- `mxalloy/py.typed` marker (README claims a typed API; PEP 561 marker missing).
- Trove classifiers (license, Python 3.11/3.12, `OS :: MacOS`, dev-status).
- Drop/relocate phantom dep `diffusers` (declared, **never imported** in `mxalloy/`); audit `safetensors`/`hf-hub`/`tqdm`/`packaging` placement.
- README's `benchmarks/benchmark_kv_cache.py` reference is a **phantom file** (only `benchmark_flux.py`, `benchmark_klein.py` exist).
- Quarantine internal planning docs (`AGENT_WORKFLOW.md`, `SCHEDULE.md`, `ARCHITECTURE_SPLIT.md`) — they reference deleted APIs and repeat the compiled-kernel framing.
- Trim/soften inert subsystems (`KernelRegistry`, `RuntimeSchedule`, `get_logger` — zero call-sites but README sells them).
- One memory figure everywhere (README 4.61 GB vs BENCHMARKS 4.54 GB).
- Reconcile README ("engine API + config + errors") vs VERSIONING ("loader + config + errors").

## Already solid — do not re-litigate

Correctly-scoped pure-Python wheel; `import mxalloy` genuinely mlx-free (PEP-562 lazy gate);
core↛models boundary + mflux-not-a-runtime-dep both hold and are tested; `load_quantized` +
streaming-load/tiled-VAE/LoRA engine real and wired; sensible dep split; no secrets/weights
committed; ruff clean on ship-able code; `examples/flux_text_to_image.py` correct.

## Open decisions (owner input needed)

1. **Attribution approach** for B1 (recommend: attribute mflux + reword docstrings).
2. **v0.1 public identity**: bless `Flux2KleinEngine` as a usable feature, or ship pure-infra?
3. **Author / Apache copyright owner** name + email for `pyproject.toml` + LICENSE.

## Release sequence

1. Licensing first (B1) — NOTICE, headers, honest docstrings.
2. Lock artifacts (B2, B3, py.typed, classifiers) — then rebuild + re-inspect wheel **and** sdist.
3. Honest, reachable public surface (B5–B8) — bless engine, wire ModelLoadError, drop dead config, Quickstart.
4. Fix README headline + quarantine internal docs (B4 + doc should-fixes).
5. Automate the gate (B9 + peak-memory regression test).
6. Tag & publish — full suite on the Apple leg, `twine check`, verify rendered PyPI page.
