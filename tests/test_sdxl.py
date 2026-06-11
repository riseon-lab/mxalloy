from __future__ import annotations

import json

from mxdiffusers.sdxl.weight_mapping import (
    remap_sdxl_text_encoder_key,
    remap_sdxl_unet_key,
    remap_sdxl_vae_key,
)


def test_unet_remap_is_identity() -> None:
    for key in (
        "conv_in.weight",
        "time_embedding.linear_1.weight",
        "add_embedding.linear_2.bias",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.weight",
        "mid_block.resnets.1.time_emb_proj.weight",
        "up_blocks.0.attentions.2.transformer_blocks.9.ff.net.0.proj.weight",
        "up_blocks.2.resnets.0.conv_shortcut.weight",
    ):
        assert remap_sdxl_unet_key(key) == key


def test_vae_remap_is_decode_only() -> None:
    assert remap_sdxl_vae_key("decoder.conv_in.weight") == "decoder.conv_in.weight"
    assert (
        remap_sdxl_vae_key("decoder.mid_block.attentions.0.to_out.0.weight")
        == "decoder.mid_block.attentions.0.to_out.0.weight"
    )
    assert remap_sdxl_vae_key("post_quant_conv.weight") == "post_quant_conv.weight"
    assert remap_sdxl_vae_key("encoder.conv_in.weight") is None
    assert remap_sdxl_vae_key("quant_conv.weight") is None


def test_text_encoder_remap_drops_position_ids() -> None:
    key = "text_model.encoder.layers.3.self_attn.q_proj.weight"
    assert remap_sdxl_text_encoder_key(key) == key
    assert remap_sdxl_text_encoder_key("text_projection.weight") == "text_projection.weight"
    assert remap_sdxl_text_encoder_key("text_model.embeddings.position_ids") is None


def test_sdxl_lora_targets_unet_linears() -> None:
    from mxdiffusers.sdxl.lora import target_paths_for_lora_base

    base = "unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
    assert target_paths_for_lora_base(base) == [
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q"
    ]
    assert target_paths_for_lora_base(
        "unet.mid_block.attentions.0.transformer_blocks.4.ff.net.2"
    ) == ["mid_block.attentions.0.transformer_blocks.4.ff.net.2"]
    # to_out -> to_out.0 normalisation
    assert target_paths_for_lora_base(
        "unet.up_blocks.0.attentions.1.transformer_blocks.2.attn2.to_out"
    ) == ["up_blocks.0.attentions.1.transformer_blocks.2.attn2.to_out.0"]
    # text-encoder and unknown keys are skipped, not crashed
    assert target_paths_for_lora_base("text_encoder.text_model.encoder.layers.0.mlp.fc1") == []
    assert target_paths_for_lora_base("unet.conv_in") == []


def test_euler_schedule_matches_diffusers_reference() -> None:
    # Reference values captured from diffusers.EulerDiscreteScheduler with the SDXL base
    # config (scaled_linear 0.00085..0.012, leading, offset 1) at num_steps=4.
    import pytest

    from tests._mlx import require_mlx_core

    require_mlx_core()
    from mxdiffusers.sdxl.scheduler import EulerDiscreteScheduler

    s = EulerDiscreteScheduler()
    timesteps, sigmas = s.make_schedule(4)
    assert [float(t) for t in timesteps] == [751.0, 501.0, 251.0, 1.0]
    expected = [4.116698, 1.623693, 0.698399, 0.041314, 0.0]
    for got, want in zip([float(x) for x in sigmas], expected, strict=True):
        assert got == pytest.approx(want, abs=2e-4)
    assert s.init_noise_sigma(sigmas) == pytest.approx(4.236414, abs=2e-4)
    import mlx.core as mx

    assert float(s.scale_model_input(mx.array(1.0), float(sigmas[0]))) == pytest.approx(
        0.236049, abs=1e-4
    )


def test_auto_detects_architectures(tmp_path) -> None:
    from mxdiffusers.auto import detect_architecture

    # canonical: model_index.json
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "ZImagePipeline"}))
    assert detect_architecture(str(tmp_path)) == "ZImagePipeline"

    # component-only snapshot: SDXL unet (text_time) vs SD1.5 unet
    sdxl = tmp_path / "sdxl"
    (sdxl / "unet").mkdir(parents=True)
    (sdxl / "unet" / "config.json").write_text(
        json.dumps({"_class_name": "UNet2DConditionModel", "addition_embed_type": "text_time"})
    )
    assert detect_architecture(str(sdxl)) == "StableDiffusionXLPipeline"

    sd15 = tmp_path / "sd15"
    (sd15 / "unet").mkdir(parents=True)
    (sd15 / "unet" / "config.json").write_text(
        json.dumps({"_class_name": "UNet2DConditionModel"})
    )
    assert detect_architecture(str(sd15)) == "StableDiffusionPipeline"

    klein = tmp_path / "klein"
    (klein / "transformer").mkdir(parents=True)
    (klein / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "Flux2Transformer2DModel"})
    )
    assert detect_architecture(str(klein)) == "Flux2Pipeline"


def test_auto_rejects_unimplemented_with_status(tmp_path) -> None:
    import pytest

    from mxalloy.errors import ModelLoadError
    from mxdiffusers.auto import MXAutoPipeline

    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusion3Pipeline"})
    )
    with pytest.raises(ModelLoadError, match="planned"):
        MXAutoPipeline.from_pretrained(str(tmp_path))


def test_flux_family_front_door_reports_flux1_status(tmp_path) -> None:
    # MXFluxPipeline is the FLUX *family* class: a FLUX.1-generation checkpoint must get a
    # planned-status error, not a klein-shaped loading crash. Detection is mlx-free.
    import pytest

    from mxalloy.errors import ModelLoadError
    from mxdiffusers.flux.pipeline import MXFluxPipeline

    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": "FluxPipeline"}))
    with pytest.raises(ModelLoadError, match="FLUX.1"):
        MXFluxPipeline.from_pretrained(str(tmp_path))
