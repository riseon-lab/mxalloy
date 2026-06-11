from __future__ import annotations

from mxdiffusers.flux.weight_mapping import (
    remap_text_encoder_key,
    remap_transformer_key,
    remap_vae_decode_key,
)


def test_transformer_remap_is_identity() -> None:
    for key in (
        "x_embedder.weight",
        "context_embedder.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight",
        "double_stream_modulation_img.linear.weight",
        "single_stream_modulation.linear.weight",
        "transformer_blocks.3.attn.to_out.0.weight",
        "transformer_blocks.3.attn.add_q_proj.weight",
        "transformer_blocks.3.attn.norm_added_k.weight",
        "transformer_blocks.3.ff_context.linear_in.weight",
        "single_transformer_blocks.17.attn.to_qkv_mlp_proj.weight",
        "single_transformer_blocks.17.attn.to_out.weight",
        "norm_out.linear.weight",
        "proj_out.weight",
    ):
        assert remap_transformer_key(key) == key


def test_text_encoder_remap_mirrors_model_subtree() -> None:
    for key in (
        "model.embed_tokens.weight",
        "model.layers.9.self_attn.q_norm.weight",
        "model.layers.9.mlp.gate_proj.weight",
        "model.norm.weight",
    ):
        assert remap_text_encoder_key(key) == key
    assert remap_text_encoder_key("lm_head.weight") is None


def test_vae_remap_is_decode_only_with_bn_stats() -> None:
    for key in (
        "decoder.conv_in.weight",
        "decoder.mid_block.attentions.0.to_out.0.weight",
        "decoder.up_blocks.2.resnets.0.conv_shortcut.weight",
        "post_quant_conv.weight",
        "bn.running_mean",
        "bn.running_var",
    ):
        assert remap_vae_decode_key(key) == key
    assert remap_vae_decode_key("bn.num_batches_tracked") is None
    assert remap_vae_decode_key("encoder.conv_in.weight") is None
    assert remap_vae_decode_key("quant_conv.weight") is None
