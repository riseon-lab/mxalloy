# v0.1 Release Checklist

When every box below is checked, mxalloy is ready to announce publicly. Items marked ☑ were
verified during the 2026-06-10 distribution-readiness pass; ☐ items are the remaining gate.

## Code quality

- ☑ Test suite green (44 tests; mlx-gated tests skip cleanly without mlx)
- ☑ `ruff check .` clean on ship-able code
- ☑ `mypy --strict` clean on the `mxalloy` core; `py.typed` shipped
- ☑ No dead public API: every exported symbol has call-sites or tests; every documented
  exception type has live raise sites (`tests/test_errors.py`)
- ☑ Architecture boundary enforced by test: `mxalloy` never imports model packages
- ☐ Decide Z-Image seed semantics before tagging: the engine seeds the global MLX RNG while
  FLUX uses a per-call key. Switching to a key changes the image produced per seed — cheapest
  now, breaking after release. (Decision owner: maintainer.)

## Packaging

- ☑ `python -m build` produces a wheel with exactly `mxalloy`/`mxdiffusers`/`mxtts` (+
  `py.typed`), no `surface/`, no internal docs; sdist matches the include list
- ☑ `twine check` passes (run in CI on every push)
- ☑ Clean-venv install of the wheel imports mlx-free from a non-repo cwd (CI `wheel-smoke`)
- ☑ Declared dependencies are all actually imported (phantom deps removed); `mlx>=0.31`
  floor matches the oldest verified version
- ☑ Classifiers + keywords present; README logo URL is absolute (renders on PyPI)
- ☐ Set the real author/maintainer name + contact email in `pyproject.toml` and confirm the
  LICENSE copyright line (currently "Alloy Contributors" placeholder)
- ☐ Reserve the `mxdiffusers` and `mxtts` names on PyPI (placeholder dists depending on
  mxalloy) so the bundled import packages can't be squatted
- ☐ Fresh `python -m build` + `twine check` + manual wheel/sdist listing immediately before
  `twine upload` (never upload a stale `dist/`)
- ☐ Verify the rendered PyPI page (readme, image, links) on test.pypi.org first

## Documentation

- ☑ README: what/who/why, measured benchmarks with hardware stated, working quickstart,
  architecture, diffusers relationship, non-goals, limitations, roadmap, contributing
- ☑ Quickstart actually works: `from_pretrained("<hf-repo-id>")` resolves the HF cache and
  the failure mode names the exact download command
- ☑ `docs/VERSIONING.md` matches the shipped public API (loader + errors + runtime planning;
  one distribution / three import packages)
- ☑ `docs/ERRORS.md` documents only exceptions that are actually raised
- ☐ Final pass: every README claim re-checked against `main` the day of release

## Benchmarks

- ☑ All published numbers come from `benchmarks/` scripts on stated hardware (18 GB M3 Pro);
  no extrapolation
- ☐ Re-run `benchmarks/benchmark_klein.py` + the mflux comparison on the release commit and
  refresh `docs/BENCHMARKS.md` dates (numbers must match the README at tag time)
- ☐ (Should-have) A peak-memory regression test: assert 4-bit load peak < 0.6× bf16 on the
  macOS CI leg — converts the headline claim into a merge gate

## Provenance / licensing

- ☑ `NOTICE` carries mflux (MIT) for `mxdiffusers/flux` and diffusers (Apache-2.0) for the
  Z-Image transformer derivation; no runtime dependency on either
- ☑ No "clean-room" overclaim anywhere: Z-Image is described as an independent MLX
  reimplementation derived from the diffusers reference (it was source-grounded)
- ☑ `PROVENANCE.md` matches the code (the only flux→zimage shared files are the Qwen3 text
  encoder and the VAE decoder helper)
- ☑ No model weights, secrets, or personal paths in the repo or artifacts

## mxdiffusers

- ☑ Both families generate on 18 GB hardware (klein 4-bit resident; Z-Image 6.2 GB)
- ☑ One `MXPipeline` contract across families (`from_pretrained`/`__call__`/LoRA trio);
  surface drives them purely polymorphically
- ☑ LoRA: hot-swap, replace-semantics, `[]` clears; failed apply cannot leave a stale
  active set (surface)
- ☐ Generate one image per family from a fresh checkout following only the README, on the
  release commit (manual gate on real hardware)

## Minimum examples (all must run as documented)

- ☑ README quickstart (FLUX + Z-Image variants)
- ☑ Runtime-direct snippet (`component_files` + `load_quantized` + missing-coverage check)
- ☐ `examples/flux_text_to_image.py` run end-to-end on release commit
- `examples/miso_text_to_speech.py` is documented as a repo-checkout spike (external
  MisoTTS clone, Python 3.10) — acceptable as-is for v0.1 if labeled, not a gate

## Known limitations (must appear in release notes verbatim-equivalent)

Apple Silicon only; measured on one 18 GB M3 Pro config; batch-1 txt2img; no img2img;
guidance inert on both shipped checkpoints; Miso TTS is an upstream-runtime spike;
mxdiffusers API stabilising (not yet under the 0.x promise).

## Release notes (draft structure)

1. What it is (two sentences) + the four headline measured numbers
2. What ships: runtime API, two image families, TTS spike, tester UI (repo-only)
3. Provenance statement (mflux port lineage, diffusers derivation, Apache-2.0)
4. Known limitations (above)
5. Roadmap pointer + how to contribute

## Announcement channels (in order)

1. GitHub: tag `v0.1.0`, release notes, repo topics (`mlx`, `apple-silicon`, `flux`,
   `diffusion`, `quantization`, `metal`)
2. PyPI publish (after test.pypi.org dry run)
3. Show HN: benchmark-led title (e.g. "mxalloy – run FLUX on a 16 GB Mac: 4.5 GB load peak,
   flat memory to 2048²"), first comment = honest provenance + limitations + how it differs
   from mflux/Draw Things
4. r/LocalLLaMA and r/StableDiffusion: the memory table + 2048²-on-18 GB images; lead with
   the problem (swap), not the project
5. MLX community: GitHub Discussions on ml-explore/mlx, Hugging Face MLX community org
6. X/LinkedIn: short thread — the load-peak chart, the flat-tiled-decode chart, repo link
7. Targeted Discords (MLX/Apple-Silicon AI, Stable Diffusion tooling) where benchmarks
   answer questions directly

No paid promotion, no cross-posting blitz: one strong technical README + reproducible
benchmark scripts is the strategy. Each post must include the reproduce-it-yourself command.
