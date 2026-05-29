# Versioning Policy

Alloy follows semantic versioning, with one explicit promise during the 0.x series.

## Public API

The public API is:

- `mxalloy.integrations.diffusers.enable_alloy`
- The config dataclasses: `AlloyConfig`, `QuantizationConfig`, `RuntimeConfig`
- The exception types in `mxalloy.errors`

Everything else is **internal** and may change in any release — including the INT8 weight format (`Int8QuantizedWeight`) and all of `mxalloy/quant`, `mxalloy/attention`, `mxalloy/kernels`, `mxalloy/runtime`, and `mxalloy/models`. Keeping the quant format internal is deliberate: calibration work (Week 6) is expected to change it.

## The 0.x Promise: Stable Within Minor

- **Patch releases (0.N.x)** never change the public API. Bug fixes and internal changes only.
- **Minor releases (0.N → 0.(N+1))** may introduce breaking changes to the public API.
- **Deprecations** are warned at least one minor release ahead. A symbol deprecated in 0.N keeps working (emitting `DeprecationWarning`) and is removed no earlier than 0.(N+1).

This is what makes the drop-in promise credible: an app pinned to `mxalloy~=0.1` won't break on a patch upgrade.

## What Counts as a Breaking Change

- Removing or renaming a public symbol.
- Changing the signature of a public function (removing/reordering parameters, or changing a default in a way that alters behavior).
- Adding a required parameter to a public function.
- Removing or renaming a field on a public dataclass, or changing its type.
- Changing the class hierarchy of public exceptions in a way that breaks existing `except` clauses.

## What Is Not a Breaking Change

- Adding a new optional parameter (with a default) to a public function.
- Adding a new public function, dataclass, or exception type.
- Adding a new field with a default to a public dataclass.
- Any change to internal modules, including the INT8 weight format.
- Performance changes, error message wording, or logging output.

## 1.0

1.0 is cut when the public API has held stable across at least two minor releases and the Phase 1 success criteria hold on reference hardware. After 1.0, breaking changes require a major version bump.
