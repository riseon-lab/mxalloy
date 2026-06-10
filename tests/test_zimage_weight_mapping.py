from __future__ import annotations

from mxdiffusers.zimage.weight_mapping import (
    remap_zimage_text_encoder_key,
    remap_zimage_transformer_key,
    remap_zimage_vae_key,
)


def test_transformer_identity_and_pad_tokens() -> None:
    assert remap_zimage_transformer_key("x_pad_token") == "x_pad_token"
    assert remap_zimage_transformer_key("cap_pad_token") == "cap_pad_token"
    assert (
        remap_zimage_transformer_key("layers.7.attention.to_out.0.weight")
        == "layers.7.attention.to_out.0.weight"
    )
    assert (
        remap_zimage_transformer_key("noise_refiner.0.adaLN_modulation.0.weight")
        == "noise_refiner.0.adaLN_modulation.0.weight"
    )
    assert (
        remap_zimage_transformer_key("context_refiner.1.attention.qkv.weight")
        == "context_refiner.1.attention.qkv.weight"
    )


def test_transformer_top_level_renames() -> None:
    assert remap_zimage_transformer_key("all_x_embedder.2-1.weight") == "x_embedder.weight"
    assert remap_zimage_transformer_key("cap_embedder.0.weight") == "cap_embedder.norm.weight"
    assert remap_zimage_transformer_key("cap_embedder.1.weight") == "cap_embedder.proj.weight"
    assert remap_zimage_transformer_key("t_embedder.mlp.0.weight") == "t_embedder.l1.weight"
    assert remap_zimage_transformer_key("t_embedder.mlp.2.bias") == "t_embedder.l2.bias"
    assert (
        remap_zimage_transformer_key("all_final_layer.2-1.linear.weight")
        == "final_layer.linear.weight"
    )
    assert (
        remap_zimage_transformer_key("all_final_layer.2-1.adaLN_modulation.1.weight")
        == "final_layer.adaLN_proj.weight"
    )


def test_transformer_drops_unknown_keys() -> None:
    # e.g. the Omni variant's siglip_* tower must be skipped, not crash the load
    assert remap_zimage_transformer_key("siglip_embedder.proj.weight") is None
    assert remap_zimage_transformer_key("unrelated.weight") is None


def test_text_encoder_strips_model_prefix() -> None:
    assert remap_zimage_text_encoder_key("model.embed_tokens.weight") == "embed_tokens.weight"
    assert (
        remap_zimage_text_encoder_key("model.layers.3.mlp.gate_proj.weight")
        == "layers.3.mlp.gate_proj.weight"
    )
    assert remap_zimage_text_encoder_key("lm_head.weight") is None


def test_vae_is_decode_only_with_to_out_collapse() -> None:
    assert remap_zimage_vae_key("decoder.conv_in.weight") == "decoder.conv_in.weight"
    assert (
        remap_zimage_vae_key("decoder.mid_block.attentions.0.to_out.0.weight")
        == "decoder.mid_block.attentions.0.to_out.weight"
    )
    assert remap_zimage_vae_key("encoder.conv_in.weight") is None
    assert remap_zimage_vae_key("quant_conv.weight") is None
