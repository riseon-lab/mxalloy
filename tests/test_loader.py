from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from mlx import nn
from mlx.utils import tree_flatten


class _Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(256, 128, bias=False)  # 2D weight, quantizable (256 % 64 == 0)


def _save(tmp_path, weights) -> list[str]:
    p = tmp_path / "w.safetensors"
    mx.save_safetensors(str(p), weights)
    return [str(p)]


def _param_keys(module) -> set[str]:
    return {k for k, _ in tree_flatten(module.parameters())}


def test_load_quantized_into_module(tmp_path):
    from mxalloy.loader import QuantConfig, load_quantized

    module = _Net()
    files = _save(tmp_path, {"lin.weight": mx.random.normal((128, 256)).astype(mx.bfloat16)})
    missing = load_quantized(module, files, quant=QuantConfig(bits=4, group_size=64))
    assert missing == set()  # full coverage
    keys = _param_keys(module)
    assert "lin.scales" in keys and "lin.biases" in keys  # weight was quantized in place


def test_load_quantized_bf16_passthrough(tmp_path):
    from mxalloy.loader import QuantConfig, load_quantized

    module = _Net()
    files = _save(tmp_path, {"lin.weight": mx.random.normal((128, 256)).astype(mx.bfloat16)})
    missing = load_quantized(module, files, quant=QuantConfig(bits=None))  # bf16, no quant
    assert missing == set()
    keys = _param_keys(module)
    assert "lin.scales" not in keys  # not quantized
    assert module.lin.weight.shape == (128, 256)


def test_load_quantized_remap_and_skip(tmp_path):
    from mxalloy.loader import QuantConfig, load_quantized

    module = _Net()
    files = _save(
        tmp_path,
        {
            "model.lin.weight": mx.random.normal((128, 256)).astype(mx.bfloat16),
            "lm_head.weight": mx.random.normal((4, 4)).astype(mx.bfloat16),  # unmapped -> skipped
        },
    )
    remap = lambda k: "lin.weight" if k == "model.lin.weight" else None  # noqa: E731
    missing = load_quantized(module, files, remap=remap, quant=QuantConfig(bits=None))
    assert missing == set()


def test_load_quantized_reports_missing(tmp_path):
    from mxalloy.loader import QuantConfig, load_quantized

    module = _Net()
    files = _save(tmp_path, {})  # nothing to load
    missing = load_quantized(module, files, quant=QuantConfig(bits=None))
    assert missing == {"lin.weight"}  # the one param went unpopulated


def test_component_files(tmp_path):
    from mxalloy.loader import component_files

    comp = tmp_path / "transformer"
    comp.mkdir()
    (comp / "a.safetensors").write_bytes(b"")
    (comp / "b.safetensors").write_bytes(b"")
    assert len(component_files(tmp_path, "transformer")) == 2
    with pytest.raises(FileNotFoundError):
        component_files(tmp_path, "missing")
