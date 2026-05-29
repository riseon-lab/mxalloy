"""Verify mxalloy's VAE decode path matches the mflux reference.

Full channel config, tiny spatial size (8x8 packed latents) to keep it fast. Copies the
decode-path weights (decoder, post_quant_conv, bn) from mflux's full VAE into our
decode-only VAE, then diffs decode_packed_latents. The encoder/quant_conv keys are
ref-only and intentionally ignored. mflux is a dev-time oracle only.

    PYTHONPATH=. .venv/bin/python experiments/verify_vae.py
"""

import mlx.core as mx
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE as RefVAE
from mlx.utils import tree_flatten, tree_unflatten

from mxalloy.models.flux2.vae import Flux2VAE as OurVAE


def main() -> None:
    mx.random.seed(0)
    ref = RefVAE()
    ours = OurVAE()

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
    ours.update(tree_unflatten(ref_flat))
    mx.eval(ref.parameters(), ours.parameters())

    packed = mx.random.normal((1, 128, 8, 8))
    ref_out = ref.decode_packed_latents(packed)
    our_out = ours.decode_packed_latents(packed)
    mx.eval(ref_out, our_out)

    print(f"decoded shape: ref {ref_out.shape}  ours {our_out.shape}")
    max_diff = float(mx.max(mx.abs(ref_out.astype(mx.float32) - our_out.astype(mx.float32))))
    print(f"max_abs_diff = {max_diff:.3e}")
    print("RESULT:", "MATCH" if max_diff < 1e-3 else "MISMATCH")


if __name__ == "__main__":
    main()
