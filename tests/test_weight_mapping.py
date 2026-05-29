from __future__ import annotations

from mxalloy.models.flux2.weight_mapping import (
    remap_text_encoder_key,
    remap_transformer_key,
    remap_vae_decode_key,
)


def test_text_encoder_strips_model_prefix() -> None:
    assert remap_text_encoder_key("model.embed_tokens.weight") == "embed_tokens.weight"
    assert (
        remap_text_encoder_key("model.layers.5.self_attn.q_proj.weight")
        == "layers.5.self_attn.q_proj.weight"
    )
    assert remap_text_encoder_key("model.norm.weight") == "norm.weight"
    assert remap_text_encoder_key("lm_head.weight") is None


def test_transformer_remaps() -> None:
    assert (
        remap_transformer_key("time_guidance_embed.timestep_embedder.linear_1.weight")
        == "time_guidance_embed.linear_1.weight"
    )
    assert (
        remap_transformer_key("transformer_blocks.3.attn.to_out.0.weight")
        == "transformer_blocks.3.attn.to_out.weight"
    )
    # identity cases (no .0 on single blocks, plain projections)
    assert remap_transformer_key("x_embedder.weight") == "x_embedder.weight"
    assert (
        remap_transformer_key("single_transformer_blocks.2.attn.to_out.weight")
        == "single_transformer_blocks.2.attn.to_out.weight"
    )


def test_vae_decode_filters_and_remaps() -> None:
    assert remap_vae_decode_key("decoder.conv_in.weight") == "decoder.conv_in.weight"
    assert (
        remap_vae_decode_key("decoder.mid_block.attentions.0.to_out.0.weight")
        == "decoder.mid_block.attentions.0.to_out.weight"
    )
    assert (
        remap_vae_decode_key("decoder.mid_block.attentions.0.to_out.0.bias")
        == "decoder.mid_block.attentions.0.to_out.bias"
    )
    assert remap_vae_decode_key("post_quant_conv.weight") == "post_quant_conv.weight"
    assert remap_vae_decode_key("bn.running_mean") == "bn.running_mean"
    # encoder / quant_conv are encode-only -> dropped
    assert remap_vae_decode_key("encoder.conv_in.weight") is None
    assert remap_vae_decode_key("quant_conv.weight") is None
