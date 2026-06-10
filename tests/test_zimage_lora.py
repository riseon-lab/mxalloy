from __future__ import annotations

from tests._mlx import require_mlx_nn


def test_zimage_lora_target_mapping_accepts_diffusers_prefixes() -> None:
    require_mlx_nn()
    from mxdiffusers.zimage.lora import target_paths_for_lora_base

    assert target_paths_for_lora_base("layers.0.attention.to_q") == [
        "layers.0.attention.to_q"
    ]
    assert target_paths_for_lora_base("transformer.layers.0.attention.to_out") == [
        "layers.0.attention.to_out.0"
    ]
    assert target_paths_for_lora_base("base_model.model.transformer.layers.0.feed_forward.w1") == [
        "layers.0.feed_forward.w1"
    ]
    assert target_paths_for_lora_base("diffusion_model.all_x_embedder.2-1") == [
        "x_embedder"
    ]
    assert target_paths_for_lora_base("all_final_layer.2-1.adaLN_modulation.1") == [
        "final_layer.adaLN_proj"
    ]
    assert target_paths_for_lora_base("cap_embedder.0") == []


def test_zimage_lora_applies_bf16_tensors_to_linear_wrapper() -> None:
    mx, nn, _tree_flatten = require_mlx_nn()
    from mxdiffusers.zimage.lora import LoRALinear, apply_loras

    class TinyAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_q = nn.Linear(4, 4, bias=False)

    class TinyLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = TinyAttention()

    class TinyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = [TinyLayer()]

    transformer = TinyTransformer()
    state = {
        "transformer.layers.0.attention.to_q.lora_A.default.weight": mx.ones(
            (2, 4), dtype=mx.bfloat16
        ),
        "transformer.layers.0.attention.to_q.lora_B.default.weight": mx.ones(
            (4, 2), dtype=mx.bfloat16
        ),
        "transformer.layers.0.attention.to_q.alpha": mx.array(2.0),
    }

    summary = apply_loras(transformer, [(state, 0.5)])

    assert summary == {"applied": 1, "skipped": []}
    assert isinstance(transformer.layers[0].attention.to_q, LoRALinear)
    y = transformer.layers[0].attention.to_q(mx.ones((1, 3, 4), dtype=mx.bfloat16))
    mx.eval(y)
    assert y.shape == (1, 3, 4)
