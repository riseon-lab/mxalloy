# Error Handling

Alloy errors should tell you what went wrong, what was expected, and what to do. Errors are part of the public API: users write `except` clauses against them, so the hierarchy is stable under the [versioning policy](VERSIONING.md).

## Exception Hierarchy

All Alloy errors derive from `AlloyError`, so `except AlloyError` catches the whole family. Where a failure is also naturally a builtin error, the type multiply-inherits from that builtin, so existing `except ValueError` / `except RuntimeError` handlers keep working.

```
AlloyError(Exception)
├── ConfigurationError(AlloyError, ValueError)
└── ModelLoadError(AlloyError, RuntimeError)
```

| Exception | Raised by |
|---|---|
| `ConfigurationError` | Invalid planning/runtime inputs: a negative `memory_budget_gb` (`detect_device_profile`), a `ComponentSpec` with neither `params` nor measured memory, an unsupported requested memory mode (`plan_execution`). |
| `ModelLoadError` | A model or its weights can't be found or loaded: no safetensors for a component (`component_files`), a file that isn't a tensor dict (`load_quantized`), an unresolvable model id (`mxdiffusers.hub.resolve_model_dir` — message includes the `huggingface-cli download` command). |

Every documented type has live raise sites; the table above is the contract, enforced by `tests/test_errors.py`. New types are added only together with the code that raises them (a `QuantizationError`, `IncompatibleLoRAError`, and `UnsupportedHardwareError` existed in pre-release drafts and were removed precisely because nothing raised them).

Import from either `mxalloy` or `mxalloy.errors`:

```python
from mxalloy import ModelLoadError
from mxalloy.errors import AlloyError
```

## Alloy Type vs Builtin

Use an Alloy exception for failures an application developer will catch and handle — bad config, load failures. These are the contract.

Plain builtins (`ValueError`, `TypeError`) remain fine for low-level programming errors in internal/utility code — e.g. a `None` passed where a value was required, or an empty-string argument. Don't dress every guard clause in a custom type.

## Message Style

Every message should answer three questions:

1. **What happened** — the specific failure, not a category.
2. **What was expected vs received** — include the offending value or shape.
3. **What to do** — the fix or next step, when there's an obvious one.

Good:

```
ModelLoadError: Tongyi-MAI/Z-Image-Turbo not found locally (not a directory, and not in the
Hugging Face cache). Download it first: huggingface-cli download Tongyi-MAI/Z-Image-Turbo
```

Avoid:

```
ValueError: bad model
```
