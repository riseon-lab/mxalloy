"""Verify mxalloy's Qwen3TextEncoder forward matches the mflux reference.

Tiny-config equivalence check: build both encoders at a small config, copy mflux's
(random) weights into ours, run identical token ids through both, and diff the final
hidden output and the stacked prompt embeds. Validates the forward math (independent of
weight loading, quantization, and tokenizer). mflux is a dev-time oracle only.

    PYTHONPATH=. .venv/bin/python experiments/verify_text_encoder.py
"""

import mlx.core as mx
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import (
    Qwen3TextEncoder as RefEncoder,
)
from mlx.utils import tree_flatten, tree_unflatten

from mxdiffusers.flux.text_encoder import Qwen3TextEncoder as OurEncoder

CFG = dict(
    vocab_size=128,
    hidden_size=16,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=32,
    head_dim=8,
    rope_theta=1000000.0,
    rms_norm_eps=1e-6,
)
LAYERS = (1, 2, 3)


def _max_diff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def main() -> None:
    mx.random.seed(0)
    ref = RefEncoder(**CFG)
    ours = OurEncoder(**CFG)
    our_keys = {k for k, _ in tree_flatten(ours.parameters())}
    ref_keys = {k for k, _ in tree_flatten(ref.parameters())}
    ref_flat = [(k, v) for k, v in tree_flatten(ref.parameters()) if k in our_keys]
    ref_only = ref_keys - our_keys
    ours_only = our_keys - ref_keys
    print(
        f"keys: ours={len(our_keys)} copied={len(ref_flat)} "
        f"ref_only={len(ref_only)} ours_only={len(ours_only)}"
    )
    if ours_only:
        print("OURS HAS KEYS REF LACKS (port bug):", sorted(ours_only)[:10])
        return
    if ref_only:
        print("ref-only (config, not weights; ignored):", sorted(ref_only)[:5])
    ours.update(tree_unflatten(ref_flat))
    mx.eval(ref.parameters(), ours.parameters())

    input_ids = mx.random.randint(0, CFG["vocab_size"], (1, 5))
    mask = mx.ones((1, 5), dtype=mx.int32)

    ref_out, _ = ref(input_ids, mask, output_hidden_states=False)
    our_out, _ = ours(input_ids, mask, output_hidden_states=False)
    mx.eval(ref_out, our_out)
    hidden_diff = _max_diff(ref_out, our_out)

    ref_pe = ref.get_prompt_embeds(input_ids, mask, hidden_state_layers=LAYERS)
    our_pe = ours.get_prompt_embeds(input_ids, mask, hidden_state_layers=LAYERS)
    mx.eval(ref_pe, our_pe)
    pe_diff = _max_diff(ref_pe, our_pe)

    print(f"hidden shape {our_out.shape}  max_abs_diff = {hidden_diff:.3e}")
    print(f"prompt_embeds shape {our_pe.shape}  max_abs_diff = {pe_diff:.3e}")
    ok = hidden_diff < 1e-3 and pe_diff < 1e-3
    print("RESULT:", "MATCH" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
