from __future__ import annotations

import numpy as np

from mxalloy.integrations.diffusers import enable_alloy
from mxalloy.quant import quantize_int8_weight
from mxalloy.runtime import detect_device


class DummyPipeline:
    pass


def test_enable_alloy_attaches_config() -> None:
    pipe = enable_alloy(DummyPipeline(), quantization="int8", loras=["adapter.safetensors"])
    assert pipe._mxalloy_config.quantization.mode == "int8"
    assert len(pipe._mxalloy_config.lora_paths) == 1


def test_quantize_int8_weight_shape() -> None:
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)
    quantized = quantize_int8_weight(weight, group_size=4)
    assert quantized.values.shape == weight.shape


def test_detect_device_returns_structured_result() -> None:
    device = detect_device()
    assert isinstance(device.machine, str)

