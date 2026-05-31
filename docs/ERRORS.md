# Error Handling

Alloy errors should tell you what went wrong, what was expected, and what to do. Errors are part of the public API: users write `except` clauses against them, so the hierarchy is stable under the [versioning policy](VERSIONING.md).

## Exception Hierarchy

All Alloy errors derive from `AlloyError`, so `except AlloyError` catches the whole family. Where a failure is also naturally a builtin error, the type multiply-inherits from that builtin, so existing `except ValueError` / `except RuntimeError` handlers keep working.

```
AlloyError(Exception)
├── ConfigurationError(AlloyError, ValueError)
├── QuantizationError(AlloyError, ValueError)
├── IncompatibleLoRAError(AlloyError, ValueError)
├── UnsupportedHardwareError(AlloyError, RuntimeError)
└── ModelLoadError(AlloyError, RuntimeError)
```

| Exception | Raise when |
|---|---|
| `ConfigurationError` | User-facing config is invalid or conflicting (missing model source, incompatible quantization + option). |
| `QuantizationError` | A weight can't be quantized/dequantized as requested (bad group size, bad shape). |
| `IncompatibleLoRAError` | A LoRA's rank, alpha, target modules, or tensor shapes don't fit the model. |
| `UnsupportedHardwareError` | The device/runtime can't run the requested path (not Apple Silicon, MLX missing). |
| `ModelLoadError` | A model or its weights can't be found or loaded. |

Import from either `mxalloy` or `mxalloy.errors`:

```python
from mxalloy import IncompatibleLoRAError
from mxalloy.errors import AlloyError
```

## Alloy Type vs Builtin

Use an Alloy exception for failures an application developer will catch and handle — bad config, incompatible LoRA, unsupported hardware, load/quant failures. These are the contract.

Plain builtins (`ValueError`, `TypeError`) remain fine for low-level programming errors in internal/utility code — e.g. a `None` passed where a value was required, or an empty-string argument. Don't dress every guard clause in a custom type.

## Message Style

Every message should answer three questions:

1. **What happened** — the specific failure, not a category.
2. **What was expected vs received** — include the offending value or shape.
3. **What to do** — the fix or next step, when there's an obvious one.

Good:

```
IncompatibleLoRAError: LoRA 'style.safetensors' targets 'transformer.x' at rank 32,
but the model exposes rank 16 for that module. Re-export the LoRA at rank <= 16, or omit it.
```

Avoid:

```
ValueError: bad lora
```

## Adoption Status

- The exception types exist and form the hierarchy above; modules raise them as they become real.
- The loader raises `ModelLoadError`; the device/runtime layer raises `UnsupportedHardwareError`; LoRA validation raises `IncompatibleLoRAError`; config validation raises `ConfigurationError`.
