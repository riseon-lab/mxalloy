"""Verify the streaming loader places real klein VAE weights correctly.

Loads the real klein VAE decode weights into both our decode-only VAE and mflux's full
VAE (via our loader), then diffs decode_packed_latents on a real-shaped latent. `missing=0`
proves our remap covers every module param; a ~0 diff proves placement + conv-transpose are
correct on real weights. mflux is a dev-time oracle only.

    PYTHONPATH=. .venv/bin/python experiments/verify_loader_vae.py
"""

import mlx.core as mx
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE as RefVAE

from mxalloy.models.flux2.loader import component_files, find_klein_model_dir, load_into_module
from mxalloy.models.flux2.vae import Flux2VAE as OurVAE
from mxalloy.models.flux2.weight_mapping import remap_vae_decode_key


def main() -> None:
    model_dir = find_klein_model_dir()
    files = component_files(model_dir, "vae")

    ours = OurVAE()
    missing = load_into_module(ours, files, remap_vae_decode_key)
    print(f"our VAE: missing params after load = {len(missing)}")
    if missing:
        print("  sample missing:", sorted(missing)[:8])

    ref = RefVAE()
    load_into_module(ref, files, remap_vae_decode_key)  # ref keeps encoder random (unused)

    packed = mx.random.normal((1, 128, 16, 16))
    our_out = ours.decode_packed_latents(packed)
    ref_out = ref.decode_packed_latents(packed)
    mx.eval(our_out, ref_out)

    diff = float(mx.max(mx.abs(our_out.astype(mx.float32) - ref_out.astype(mx.float32))))
    finite = bool(mx.all(mx.isfinite(our_out)))
    print(f"decoded {our_out.shape}  decode diff={diff:.3e}  finite={finite}")
    print("RESULT:", "MATCH" if (len(missing) == 0 and diff < 1e-3 and finite) else "MISMATCH")


if __name__ == "__main__":
    main()
