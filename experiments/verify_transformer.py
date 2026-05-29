"""Verify mxalloy's Flux2Transformer forward matches the mflux reference.

Tiny-config equivalence check: build both transformers at a small config, copy mflux's
(random) weights into ours so they are identical, then run identical inputs through both
and diff the outputs. This validates the forward math (independent of weight loading and
quantization) with negligible memory. Copying weights only works if every parameter key
aligns, so this also catches structural drift in the port.

mflux is a dev-time oracle only; it is never a dependency of the mxalloy package.

    PYTHONPATH=. .venv/bin/python experiments/verify_transformer.py
"""

import mlx.core as mx
from mflux.models.flux2.model.flux2_transformer.transformer import (
    Flux2Transformer as RefTransformer,
)
from mlx.utils import tree_flatten

from mxalloy.models.flux2.transformer import Flux2Transformer as OurTransformer

CFG = dict(
    in_channels=16,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=64,
    timestep_guidance_channels=16,
    axes_dims_rope=(2, 2, 2, 2),
    rope_theta=2000,
    guidance_embeds=False,
)


def main() -> None:
    mx.random.seed(0)
    ref = RefTransformer(**CFG)
    ours = OurTransformer(**CFG)
    ours.update(ref.parameters())
    mx.eval(ref.parameters(), ours.parameters())

    ref_keys = {k for k, _ in tree_flatten(ref.parameters())}
    our_keys = {k for k, _ in tree_flatten(ours.parameters())}
    symdiff = ref_keys ^ our_keys
    print(f"param keys: ref={len(ref_keys)} ours={len(our_keys)} symdiff={len(symdiff)}")
    if symdiff:
        print("KEY MISMATCH (first 10):", sorted(symdiff)[:10])
        return

    batch, img_seq, txt_seq = 1, 4, 3
    inputs = dict(
        hidden_states=mx.random.normal((batch, img_seq, CFG["in_channels"])),
        encoder_hidden_states=mx.random.normal((batch, txt_seq, CFG["joint_attention_dim"])),
        timestep=mx.array(0.5),
        img_ids=mx.random.uniform(0, 10, (img_seq, 4)),
        txt_ids=mx.random.uniform(0, 10, (txt_seq, 4)),
        guidance=None,
    )

    out_ref = ref(**inputs)
    out_ours = ours(**inputs)
    mx.eval(out_ref, out_ours)

    print("shape: ref", out_ref.shape, "ours", out_ours.shape)
    max_diff = float(mx.max(mx.abs(out_ref.astype(mx.float32) - out_ours.astype(mx.float32))))
    print(f"max_abs_diff = {max_diff:.3e}")
    print("RESULT:", "MATCH" if max_diff < 1e-3 else "MISMATCH")


if __name__ == "__main__":
    main()
