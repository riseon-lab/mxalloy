"""Phase 2 performance measurements for FLUX.2-klein on MLX/mxalloy.

This is intentionally measurement-only. It does not patch model code or change kernels.

Examples:
    PYTHONPATH=. .venv/bin/python experiments/perf_phase2.py e2e --bits 4
    PYTHONPATH=. .venv/bin/python experiments/perf_phase2.py timeline --bits 4
    PYTHONPATH=. .venv/bin/python experiments/perf_phase2.py linears --bits 4
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx import nn

from mxalloy.models.flux2.engine import Flux2KleinEngine
from mxalloy.models.flux2.latents import prepare_packed_latents, prepare_text_ids
from mxalloy.models.flux2.scheduler import FlowMatchEulerScheduler
from mxalloy.models.flux2.transformer import PRECISION, _apply_rope_bshd

PROMPT = "a brushed alloy sculpture under studio light"
SEED = 42
TEXT_ENCODER_OUT_LAYERS = (9, 18, 27)


def now() -> float:
    return time.perf_counter()


def ms(seconds: float) -> float:
    return seconds * 1000.0


def gb(nbytes: int | float) -> float:
    return float(nbytes) / 1024**3


def array_nbytes(x: mx.array | None) -> int:
    if x is None:
        return 0
    if hasattr(x, "nbytes"):
        return int(x.nbytes)
    size = 1
    for d in x.shape:
        size *= int(d)
    return size * int(x.dtype.size)


def tree_arrays(x: Any) -> list[mx.array]:
    out: list[mx.array] = []
    if isinstance(x, mx.array):
        out.append(x)
    elif isinstance(x, dict):
        for v in x.values():
            out.extend(tree_arrays(v))
    elif isinstance(x, (list, tuple)):
        for v in x:
            out.extend(tree_arrays(v))
    return out


def eval_tree(x: Any) -> None:
    arrays = tree_arrays(x)
    if arrays:
        mx.eval(*arrays)


def memory_snapshot() -> dict[str, float | None]:
    try:
        return {
            "active_gb": gb(mx.get_active_memory()),
            "peak_gb": gb(mx.get_peak_memory()),
            "cache_gb": gb(mx.get_cache_memory()),
        }
    except Exception:
        return {"active_gb": None, "peak_gb": None, "cache_gb": None}


def reset_peak_memory() -> None:
    try:
        mx.reset_peak_memory()
    except Exception:
        pass


def clear_cache() -> None:
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


@dataclass
class Row:
    label: str
    group: str
    graph_ms: float
    sync_ms: float
    total_ms: float
    flops: float | None = None
    lower_bound_bytes: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Recorder:
    def __init__(self) -> None:
        self.rows: list[Row] = []

    def time_eval(
        self,
        label: str,
        group: str,
        fn: Callable[[], Any],
        *,
        flops: float | None = None,
        lower_bound_bytes: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> Any:
        t0 = now()
        out = fn()
        t1 = now()
        eval_tree(out)
        t2 = now()
        self.rows.append(
            Row(
                label=label,
                group=group,
                graph_ms=ms(t1 - t0),
                sync_ms=ms(t2 - t1),
                total_ms=ms(t2 - t0),
                flops=flops,
                lower_bound_bytes=lower_bound_bytes,
                meta=meta or {},
            )
        )
        return out

    def add(
        self,
        label: str,
        group: str,
        graph_s: float,
        sync_s: float,
        total_s: float,
        *,
        flops: float | None = None,
        lower_bound_bytes: int | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.rows.append(
            Row(
                label=label,
                group=group,
                graph_ms=ms(graph_s),
                sync_ms=ms(sync_s),
                total_ms=ms(total_s),
                flops=flops,
                lower_bound_bytes=lower_bound_bytes,
                meta=meta or {},
            )
        )

    def by_group(self) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, float]] = defaultdict(
            lambda: {"graph_ms": 0.0, "sync_ms": 0.0, "total_ms": 0.0, "flops": 0.0, "bytes": 0.0}
        )
        for row in self.rows:
            g = groups[row.group]
            g["graph_ms"] += row.graph_ms
            g["sync_ms"] += row.sync_ms
            g["total_ms"] += row.total_ms
            if row.flops is not None:
                g["flops"] += row.flops
            if row.lower_bound_bytes is not None:
                g["bytes"] += row.lower_bound_bytes
        out = []
        for group, vals in groups.items():
            total_s = vals["total_ms"] / 1000.0
            out.append(
                {
                    "group": group,
                    "graph_ms": vals["graph_ms"],
                    "sync_ms": vals["sync_ms"],
                    "total_ms": vals["total_ms"],
                    "logical_tflops_per_s": vals["flops"] / total_s / 1e12
                    if total_s > 0 and vals["flops"]
                    else None,
                    "lower_bound_gb_per_s": vals["bytes"] / total_s / 1024**3
                    if total_s > 0 and vals["bytes"]
                    else None,
                }
            )
        return sorted(out, key=lambda x: x["total_ms"], reverse=True)

    def to_json(self) -> dict[str, Any]:
        return {
            "rows": [
                {
                    "label": r.label,
                    "group": r.group,
                    "graph_ms": r.graph_ms,
                    "sync_ms": r.sync_ms,
                    "total_ms": r.total_ms,
                    "flops": r.flops,
                    "lower_bound_bytes": r.lower_bound_bytes,
                    "meta": r.meta,
                }
                for r in self.rows
            ],
            "groups": self.by_group(),
        }


@dataclass
class Prepared:
    engine: Flux2KleinEngine
    latents: mx.array
    latent_ids: mx.array
    prompt_embeds: mx.array
    text_ids: mx.array
    scheduler: FlowMatchEulerScheduler
    timestep: mx.array
    height: int
    width: int


def parse_bits(value: str) -> int | None:
    v = value.lower()
    if v in {"none", "bf16", "fp16"}:
        return None
    if v in {"4", "int4"}:
        return 4
    if v in {"8", "int8"}:
        return 8
    raise ValueError(f"unsupported bits value {value!r}")


def bits_label(bits: int | None) -> str:
    return "bf16" if bits is None else f"int{bits}"


def prepare(bits: int | None, height: int, width: int) -> Prepared:
    clear_cache()
    reset_peak_memory()
    t0 = now()
    engine = Flux2KleinEngine(quantize_bits=bits)
    mx.eval(
        engine.transformer.parameters(),
        engine.text_encoder.parameters(),
        engine.vae.parameters(),
    )
    load_s = now() - t0

    input_ids, attn_mask = engine.tokenizer.encode(PROMPT)
    prompt_embeds = engine.text_encoder.get_prompt_embeds(
        input_ids,
        attn_mask,
        TEXT_ENCODER_OUT_LAYERS,
    )
    text_ids = prepare_text_ids(prompt_embeds)
    latents, latent_ids, _lh, _lw = prepare_packed_latents(
        seed=SEED, height=height, width=width, batch_size=1
    )
    scheduler = FlowMatchEulerScheduler(
        num_inference_steps=4,
        image_seq_len=(height // 16) * (width // 16),
    )
    eval_tree((prompt_embeds, text_ids, latents, latent_ids, scheduler.timesteps, scheduler.sigmas))
    prep = Prepared(
        engine=engine,
        latents=latents,
        latent_ids=latent_ids,
        prompt_embeds=prompt_embeds,
        text_ids=text_ids,
        scheduler=scheduler,
        timestep=scheduler.timesteps[0],
        height=height,
        width=width,
    )
    print_json(
        {
            "event": "prepared",
            "bits": bits_label(bits),
            "height": height,
            "width": width,
            "load_seconds": load_s,
            "memory": memory_snapshot(),
        }
    )
    return prep


def transformer_step(prep: Prepared, latents: mx.array | None = None) -> mx.array:
    return prep.engine.transformer(
        hidden_states=latents if latents is not None else prep.latents,
        encoder_hidden_states=prep.prompt_embeds,
        timestep=prep.timestep,
        img_ids=prep.latent_ids,
        txt_ids=prep.text_ids,
        guidance=None,
    )


def denoise_step(prep: Prepared, latents: mx.array | None = None) -> mx.array:
    noise = transformer_step(prep, latents)
    return prep.scheduler.step(noise, 0, latents if latents is not None else prep.latents)


def bench_e2e(prep: Prepared, iters: int, warmup: int) -> dict[str, Any]:
    for _ in range(warmup):
        eval_tree(denoise_step(prep))
    rows = []
    latents = prep.latents
    for _ in range(iters):
        reset_peak_memory()
        t0 = now()
        noise = transformer_step(prep, latents)
        t1 = now()
        next_latents = prep.scheduler.step(noise, 0, latents)
        t2 = now()
        eval_tree(next_latents)
        t3 = now()
        rows.append(
            {
                "transformer_graph_ms": ms(t1 - t0),
                "scheduler_graph_ms": ms(t2 - t1),
                "eval_sync_ms": ms(t3 - t2),
                "total_ms": ms(t3 - t0),
                "memory": memory_snapshot(),
            }
        )
        latents = next_latents
    med = {
        key: statistics.median(row[key] for row in rows)
        for key in ("transformer_graph_ms", "scheduler_graph_ms", "eval_sync_ms", "total_ms")
    }
    return {"iterations": rows, "median": med}


def flops_linear(x: mx.array, out_dims: int, in_dims: int | None = None) -> float:
    m = 1
    for d in x.shape[:-1]:
        m *= int(d)
    k = int(in_dims or x.shape[-1])
    n = int(out_dims)
    return 2.0 * m * k * n


def quantized_dims(layer: nn.Module) -> tuple[int, int, int | None]:
    weight = layer["weight"]
    out_dims = int(weight.shape[0])
    bits = getattr(layer, "bits", None)
    if bits is None:
        return out_dims, int(weight.shape[1]), None
    return out_dims, int(weight.shape[1]) * 32 // int(bits), int(bits)


def layer_lower_bound_bytes(layer: nn.Module, x: mx.array, y: mx.array | None = None) -> int:
    total = array_nbytes(x) + array_nbytes(y)
    for key in ("weight", "scales", "biases", "bias"):
        if key in layer:
            total += array_nbytes(layer[key])
    return total


def run_timeline(prep: Prepared) -> dict[str, Any]:
    rec = Recorder()
    tf = prep.engine.transformer
    hidden_states = prep.latents
    encoder_hidden_states = prep.prompt_embeds
    timestep = prep.timestep
    guidance = None
    measured_start = now()

    if not isinstance(timestep, mx.array):
        timestep = mx.array(timestep, dtype=hidden_states.dtype)
    if timestep.ndim == 0:
        timestep = mx.full((hidden_states.shape[0],), timestep, dtype=hidden_states.dtype)
    timestep = timestep.astype(hidden_states.dtype)
    timestep_scale = mx.where(mx.max(timestep) <= 1.0, 1000.0, 1.0).astype(hidden_states.dtype)
    timestep = rec.time_eval(
        "timestep_scale",
        "host_orchestration",
        lambda: timestep * timestep_scale,
    )

    temb = rec.time_eval(
        "time_guidance_embed",
        "conditioning",
        lambda: tf.time_guidance_embed(timestep, guidance).astype(PRECISION),
    )
    hidden_states = rec.time_eval(
        "x_embedder",
        "input_projection",
        lambda: tf.x_embedder(hidden_states),
        flops=flops_linear(hidden_states, tf.inner_dim),
    )
    encoder_hidden_states = rec.time_eval(
        "context_embedder",
        "input_projection",
        lambda: tf.context_embedder(encoder_hidden_states),
        flops=flops_linear(encoder_hidden_states, tf.inner_dim),
    )

    img_ids = prep.latent_ids[0] if prep.latent_ids.ndim == 3 else prep.latent_ids
    txt_ids = prep.text_ids[0] if prep.text_ids.ndim == 3 else prep.text_ids
    if (
        getattr(tf, "_cached_rope", None) is not None
        and getattr(tf, "_cached_rope_keys", None) is not None
        and tf._cached_rope_keys[0] is img_ids
        and tf._cached_rope_keys[1] is txt_ids
    ):
        concat_rotary_emb = tf._cached_rope
        rec.add("rope_cache_hit", "rope", 0.0, 0.0, 0.0, meta={"cached": True})
    else:
        image_rotary_emb = rec.time_eval("pos_embed_img", "rope", lambda: tf.pos_embed(img_ids))
        text_rotary_emb = rec.time_eval("pos_embed_txt", "rope", lambda: tf.pos_embed(txt_ids))
        concat_rotary_emb = rec.time_eval(
            "rope_concat",
            "rope",
            lambda: (
                mx.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
                mx.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
            ),
        )
        tf._cached_rope = concat_rotary_emb
        tf._cached_rope_keys = (img_ids, txt_ids)

    temb_mod_params_img = rec.time_eval(
        "double_stream_modulation_img",
        "conditioning",
        lambda: tf.double_stream_modulation_img(temb),
    )
    temb_mod_params_txt = rec.time_eval(
        "double_stream_modulation_txt",
        "conditioning",
        lambda: tf.double_stream_modulation_txt(temb),
    )

    for i, block in enumerate(tf.transformer_blocks):
        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = temb_mod_params_img
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = (
            temb_mod_params_txt
        )
        norm_hidden_states = rec.time_eval(
            f"double_{i}.norm1_img_mod",
            "double_norm_mod",
            lambda hs=hidden_states: (1 + scale_msa) * block.norm1(hs) + shift_msa,
        )
        norm_encoder_hidden_states = rec.time_eval(
            f"double_{i}.norm1_txt_mod",
            "double_norm_mod",
            lambda ehs=encoder_hidden_states: (1 + c_scale_msa) * block.norm1_context(ehs)
            + c_shift_msa,
        )
        attn_output, encoder_attn_output = rec.time_eval(
            f"double_{i}.attention",
            "double_attention",
            lambda nhs=norm_hidden_states, nehs=norm_encoder_hidden_states: block.attn(
                hidden_states=nhs,
                encoder_hidden_states=nehs,
                image_rotary_emb=concat_rotary_emb,
            ),
        )
        hidden_states = rec.time_eval(
            f"double_{i}.attn_residual_img",
            "residual_gate",
            lambda hs=hidden_states, ao=attn_output: hs + gate_msa * ao,
        )
        encoder_hidden_states = rec.time_eval(
            f"double_{i}.attn_residual_txt",
            "residual_gate",
            lambda ehs=encoder_hidden_states, eao=encoder_attn_output: ehs + c_gate_msa * eao,
        )
        norm_hidden_states = rec.time_eval(
            f"double_{i}.norm2_img_mod",
            "double_norm_mod",
            lambda hs=hidden_states: (1 + scale_mlp) * block.norm2(hs) + shift_mlp,
        )
        ff_img = rec.time_eval(
            f"double_{i}.ff_img",
            "double_mlp",
            lambda nhs=norm_hidden_states: block.ff(nhs),
        )
        hidden_states = rec.time_eval(
            f"double_{i}.ff_residual_img",
            "residual_gate",
            lambda hs=hidden_states, ff=ff_img: hs + gate_mlp * ff,
        )
        norm_encoder_hidden_states = rec.time_eval(
            f"double_{i}.norm2_txt_mod",
            "double_norm_mod",
            lambda ehs=encoder_hidden_states: (1 + c_scale_mlp) * block.norm2_context(ehs)
            + c_shift_mlp,
        )
        ff_txt = rec.time_eval(
            f"double_{i}.ff_txt",
            "double_mlp",
            lambda nehs=norm_encoder_hidden_states: block.ff_context(nehs),
        )
        encoder_hidden_states = rec.time_eval(
            f"double_{i}.ff_residual_txt",
            "residual_gate",
            lambda ehs=encoder_hidden_states, ff=ff_txt: ehs + c_gate_mlp * ff,
        )

    hidden_states = rec.time_eval(
        "concat_text_image",
        "layout_conversion",
        lambda: mx.concatenate([encoder_hidden_states, hidden_states], axis=1),
    )
    temb_mod_params_single = rec.time_eval(
        "single_stream_modulation",
        "conditioning",
        lambda: tf.single_stream_modulation(temb)[0],
    )
    mod_shift, mod_scale, mod_gate = temb_mod_params_single
    for i, block in enumerate(tf.single_transformer_blocks):
        norm_hidden_states = rec.time_eval(
            f"single_{i}.norm_mod",
            "single_norm_mod",
            lambda hs=hidden_states: (1 + mod_scale) * block.norm(hs) + mod_shift,
        )
        attn_output = rec.time_eval(
            f"single_{i}.parallel_attn_mlp",
            "single_parallel_attn_mlp",
            lambda nhs=norm_hidden_states: block.attn(nhs, concat_rotary_emb),
        )
        hidden_states = rec.time_eval(
            f"single_{i}.residual",
            "residual_gate",
            lambda hs=hidden_states, ao=attn_output: hs + mod_gate * ao,
        )

    image_tokens = prep.prompt_embeds.shape[1]
    hidden_states = rec.time_eval(
        "drop_text_tokens",
        "layout_conversion",
        lambda: hidden_states[:, image_tokens:, ...],
    )
    hidden_states = rec.time_eval(
        "norm_out",
        "output_projection",
        lambda: tf.norm_out(hidden_states, temb),
    )
    _out = rec.time_eval(
        "proj_out",
        "output_projection",
        lambda: tf.proj_out(hidden_states),
        flops=flops_linear(hidden_states, tf.out_channels),
    )
    measured_end = now()
    return {
        "measured_total_ms": ms(measured_end - measured_start),
        "timeline": rec.to_json(),
        "memory": memory_snapshot(),
    }


DeepDiveInput = tuple[nn.Module, mx.array, Callable[[mx.array], mx.array] | None]


def build_deep_dive_inputs(prep: Prepared) -> dict[str, DeepDiveInput]:
    tf = prep.engine.transformer
    hidden_states = tf.x_embedder(prep.latents)
    encoder_hidden_states = tf.context_embedder(prep.prompt_embeds)
    timestep = prep.timestep
    if timestep.ndim == 0:
        timestep = mx.full((prep.latents.shape[0],), timestep, dtype=prep.latents.dtype)
    temb = tf.time_guidance_embed(timestep.astype(prep.latents.dtype), None).astype(PRECISION)
    img_ids = prep.latent_ids[0] if prep.latent_ids.ndim == 3 else prep.latent_ids
    txt_ids = prep.text_ids[0] if prep.text_ids.ndim == 3 else prep.text_ids
    image_rotary_emb = tf.pos_embed(img_ids)
    text_rotary_emb = tf.pos_embed(txt_ids)
    concat_rotary_emb = (
        mx.concatenate([text_rotary_emb[0], image_rotary_emb[0]], axis=0),
        mx.concatenate([text_rotary_emb[1], image_rotary_emb[1]], axis=0),
    )
    temb_mod_params_img = tf.double_stream_modulation_img(temb)
    temb_mod_params_txt = tf.double_stream_modulation_txt(temb)
    eval_tree(
        (
            hidden_states,
            encoder_hidden_states,
            temb,
            concat_rotary_emb,
            temb_mod_params_img,
            temb_mod_params_txt,
        )
    )

    block = tf.transformer_blocks[0]
    (shift_msa, scale_msa, _gate_msa), (shift_mlp, scale_mlp, _gate_mlp) = temb_mod_params_img
    norm_hidden_states = (1 + scale_msa) * block.norm1(hidden_states) + shift_msa
    norm_hidden_states2 = (1 + scale_mlp) * block.norm2(hidden_states) + shift_mlp
    eval_tree((norm_hidden_states, norm_hidden_states2))

    single_block = tf.single_transformer_blocks[0]
    first_ehs, first_hs = block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb_mod_params_img=temb_mod_params_img,
        temb_mod_params_txt=temb_mod_params_txt,
        image_rotary_emb=concat_rotary_emb,
    )
    h_concat = mx.concatenate([first_ehs, first_hs], axis=1)
    temb_mod_params_single = tf.single_stream_modulation(temb)[0]
    mod_shift, mod_scale, _mod_gate = temb_mod_params_single
    norm_h_concat = (1 + mod_scale) * single_block.norm(h_concat) + mod_shift
    eval_tree((h_concat, norm_h_concat))

    return {
        "double0.attn.to_q": (block.attn.to_q, norm_hidden_states, None),
        "double0.attn.to_out": (block.attn.to_out, norm_hidden_states, None),
        "double0.ff.linear_in": (block.ff.linear_in, norm_hidden_states2, block.ff.act),
        "double0.ff.linear_out": (
            block.ff.linear_out,
            block.ff.act(block.ff.linear_in(norm_hidden_states2)),
            None,
        ),
        "single0.to_qkv_mlp_proj": (single_block.attn.to_qkv_mlp_proj, norm_h_concat, None),
        "single0.to_out": (
            single_block.attn.to_out,
            mx.zeros(
                (
                    1,
                    prep.prompt_embeds.shape[1] + prep.latents.shape[1],
                    single_block.attn.inner_dim + single_block.attn.mlp_hidden_dim,
                ),
                dtype=PRECISION,
            ),
            None,
        ),
        "output.proj_out": (tf.proj_out, first_hs, None),
    }


def median_time(fn: Callable[[], Any], iters: int = 5, warmup: int = 1) -> float:
    for _ in range(warmup):
        eval_tree(fn())
    vals = []
    for _ in range(iters):
        t0 = now()
        out = fn()
        t1 = now()
        eval_tree(out)
        t2 = now()
        vals.append(t2 - t0)
    return statistics.median(vals)


def run_linear_deep_dive(prep: Prepared, iters: int) -> dict[str, Any]:
    results = []
    inputs = build_deep_dive_inputs(prep)
    for name, (layer, x, act_fn) in inputs.items():
        y = layer(x)
        eval_tree(y)
        out_dims, in_dims, bits = quantized_dims(layer)
        flops = flops_linear(x, out_dims, in_dims)
        qmm_s = median_time(lambda l=layer, a=x: l(a), iters=iters)
        row: dict[str, Any] = {
            "name": name,
            "input_shape": list(x.shape),
            "output_shape": list(y.shape),
            "bits": bits_label(bits),
            "logical_tflops": flops / 1e12,
            "qmm_ms": ms(qmm_s),
            "qmm_tflops_per_s": flops / qmm_s / 1e12,
            "lower_bound_gb": gb(layer_lower_bound_bytes(layer, x, y)),
            "lower_bound_gb_per_s": gb(layer_lower_bound_bytes(layer, x, y)) / qmm_s,
        }
        if bits is not None:
            deq = lambda l=layer: mx.dequantize(
                l["weight"],
                l["scales"],
                l.get("biases"),
                group_size=l.group_size,
                bits=l.bits,
                mode=l.mode,
                dtype=PRECISION,
            )
            deq_weight = deq()
            eval_tree(deq_weight)
            deq_s = median_time(deq, iters=iters)
            dense_s = median_time(lambda a=x, w=deq_weight: a @ mx.transpose(w), iters=iters)
            fused_deq_dense_s = median_time(
                lambda a=x, l=layer: a
                @ mx.transpose(
                    mx.dequantize(
                        l["weight"],
                        l["scales"],
                        l.get("biases"),
                        group_size=l.group_size,
                        bits=l.bits,
                        mode=l.mode,
                        dtype=PRECISION,
                    )
                ),
                iters=iters,
            )
            row.update(
                {
                    "weight_dequant_ms": ms(deq_s),
                    "dense_bf16_matmul_ms": ms(dense_s),
                    "dense_bf16_tflops_per_s": flops / dense_s / 1e12,
                    "dequant_plus_dense_ms": ms(fused_deq_dense_s),
                }
            )
        if act_fn is not None:
            proj = layer(x)
            eval_tree(proj)
            act_s = median_time(lambda p=proj, f=act_fn: f(p), iters=iters)
            row["activation_ms"] = ms(act_s)
        results.append(row)
    return {"linears": sorted(results, key=lambda r: r["qmm_ms"], reverse=True)}


def print_json(obj: dict[str, Any]) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True), flush=True)


def write_result(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    payload = {
        "command": args.command,
        "bits": bits_label(args.bits),
        "height": args.height,
        "width": args.width,
        "device": mx.device_info() if mx.metal.is_available() else {},
        **payload,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print_json(payload)


def command_e2e(args: argparse.Namespace) -> None:
    prep = prepare(args.bits, args.height, args.width)
    result = bench_e2e(prep, args.iters, args.warmup)
    write_result(args, result)


def command_timeline(args: argparse.Namespace) -> None:
    prep = prepare(args.bits, args.height, args.width)
    for _ in range(args.warmup):
        eval_tree(denoise_step(prep))
    result = run_timeline(prep)
    # Include scheduler as a measured end-to-end tail after instrumented transformer output.
    noise = transformer_step(prep)
    eval_tree(noise)
    t0 = now()
    latents = prep.scheduler.step(noise, 0, prep.latents)
    t1 = now()
    eval_tree(latents)
    t2 = now()
    result["scheduler_tail"] = {
        "graph_ms": ms(t1 - t0),
        "sync_ms": ms(t2 - t1),
        "total_ms": ms(t2 - t0),
    }
    write_result(args, result)


def command_linears(args: argparse.Namespace) -> None:
    prep = prepare(args.bits, args.height, args.width)
    for _ in range(args.warmup):
        eval_tree(denoise_step(prep))
    result = run_linear_deep_dive(prep, args.iters)
    write_result(args, result)


def command_capture_step(args: argparse.Namespace) -> None:
    prep = prepare(args.bits, args.height, args.width)
    for _ in range(args.warmup):
        eval_tree(denoise_step(prep))
    print(f"READY pid={os.getpid()}", flush=True)
    if args.sleep_before:
        time.sleep(args.sleep_before)
    reset_peak_memory()
    t0 = now()
    latents = denoise_step(prep)
    eval_tree(latents)
    total_s = now() - t0
    print_json(
        {
            "event": "captured_step_done",
            "total_ms": ms(total_s),
            "memory": memory_snapshot(),
        }
    )
    if args.sleep_after:
        time.sleep(args.sleep_after)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--bits", type=parse_bits, default=4)
    parser.add_argument("--out")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("e2e")
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.set_defaults(func=command_e2e)

    p = sub.add_parser("timeline")
    p.add_argument("--warmup", type=int, default=1)
    p.set_defaults(func=command_timeline)

    p = sub.add_parser("linears")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.set_defaults(func=command_linears)

    p = sub.add_parser("capture-step")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--sleep-before", type=float, default=0.0)
    p.add_argument("--sleep-after", type=float, default=0.0)
    p.set_defaults(func=command_capture_step)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
