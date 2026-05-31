import time
import statistics
import mlx.core as mx
from mxalloy.models.flux2.engine import Flux2KleinEngine
from mxalloy.models.flux2.latents import prepare_packed_latents, prepare_text_ids
from mxalloy.models.flux2.scheduler import FlowMatchEulerScheduler

PROMPT = "a brushed alloy sculpture under studio light"
SEED = 42

def bench(thunk, name, iters=3, warmup=1):
    for _ in range(warmup):
        mx.eval(thunk())
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        mx.eval(thunk())
        ts.append(time.perf_counter() - t)
    median_t = statistics.median(ts)
    print(f"{name:35s}: {median_t * 1000:8.1f} ms")
    return median_t

def main():
    h, w = 1024, 1024
    engine = Flux2KleinEngine(quantize_bits=4)
    tf = engine.transformer

    input_ids, attn_mask = engine.tokenizer.encode(PROMPT)
    prompt_embeds = engine.text_encoder.get_prompt_embeds(input_ids, attn_mask, (9, 18, 27))
    text_ids = prepare_text_ids(prompt_embeds)
    latents, latent_ids, lh, lw = prepare_packed_latents(seed=SEED, height=h, width=w, batch_size=1)
    sched = FlowMatchEulerScheduler(num_inference_steps=4, image_seq_len=(h // 16) * (w // 16))
    t0 = sched.timesteps[0]

    # Warmup the whole transformer first
    mx.eval(tf(latents, prompt_embeds, t0, latent_ids, text_ids, None))

    # Break down parts
    # 1. RoPE Pos Embed calculation
    bench(lambda: tf.pos_embed(latent_ids[0] if latent_ids.ndim == 3 else latent_ids), "pos_embed(img_ids)")
    bench(lambda: tf.pos_embed(text_ids[0] if text_ids.ndim == 3 else text_ids), "pos_embed(txt_ids)")

    image_rotary_emb = tf.pos_embed(latent_ids[0])
    text_rotary_emb = tf.pos_embed(text_ids[0])
    concat_rotary_emb = (
        mx.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
        mx.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
    )

    # 2. Embedders
    bench(lambda: tf.x_embedder(latents), "x_embedder")
    bench(lambda: tf.context_embedder(prompt_embeds), "context_embedder")

    # Let's get embedded states
    hs = tf.x_embedder(latents)
    ehs = tf.context_embedder(prompt_embeds)
    temb = tf.time_guidance_embed(mx.full((1,), t0, dtype=latents.dtype), None).astype(mx.bfloat16)

    # 3. Modulation
    bench(lambda: tf.double_stream_modulation_img(temb), "double_stream_modulation_img")

    temb_mod_params_img = tf.double_stream_modulation_img(temb)
    temb_mod_params_txt = tf.double_stream_modulation_txt(temb)

    # 4. First Double Stream block
    block = tf.transformer_blocks[0]
    bench(
        lambda: block(hs, ehs, temb_mod_params_img, temb_mod_params_txt, concat_rotary_emb),
        "first double_stream block"
    )

    # Let's profile the components of the first double block
    (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
    (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = temb_mod_params_txt

    norm_hidden_states = block.norm1(hs)
    norm_hidden_states = (1 + scale_msa) * norm_hidden_states + shift_msa
    norm_encoder_hidden_states = block.norm1_context(ehs)
    norm_encoder_hidden_states = (1 + c_scale_msa) * norm_encoder_hidden_states + c_shift_msa

    bench(lambda: block.norm1(hs), "block.norm1")
    bench(lambda: block.norm1_context(ehs), "block.norm1_context")
    bench(
        lambda: block.attn(norm_hidden_states, norm_encoder_hidden_states, concat_rotary_emb),
        "block.attn (double block attention)"
    )

    # Inside block.attn:
    attn = block.attn
    bench(lambda: attn.to_q(norm_hidden_states), "attn.to_q (linear)")
    bench(lambda: attn.norm_q(attn.to_q(norm_hidden_states).reshape(1, 4096, 24, 128).transpose(0, 2, 1, 3).astype(mx.float32)), "attn.norm_q (RMSNorm)")

    # 5. First Single Stream block
    single_block = tf.single_transformer_blocks[0]
    ehs_out, hs_out = block(hs, ehs, temb_mod_params_img, temb_mod_params_txt, concat_rotary_emb)
    h_concat = mx.concatenate([ehs_out, hs_out], axis=1)
    temb_mod_params_single = tf.single_stream_modulation(temb)[0]

    bench(
        lambda: single_block(h_concat, temb_mod_params_single, concat_rotary_emb),
        "first single_stream block"
    )

    # Components of single block
    mod_shift, mod_scale, mod_gate = temb_mod_params_single
    norm_h_concat = single_block.norm(h_concat)
    norm_h_concat = (1 + mod_scale) * norm_h_concat + mod_shift

    bench(lambda: single_block.norm(h_concat), "single_block.norm")
    bench(
        lambda: single_block.attn(norm_h_concat, concat_rotary_emb),
        "single_block.attn"
    )

    # Inside single_block.attn
    sa = single_block.attn
    bench(lambda: sa.to_qkv_mlp_proj(norm_h_concat), "sa.to_qkv_mlp_proj (large fused linear)")

if __name__ == "__main__":
    main()
