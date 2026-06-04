"""Verify the streaming loader places real klein VAE weights correctly.

Loads the real klein VAE decode weights into both our decode-only VAE and mflux's full
VAE (via our loader), then diffs decode_packed_latents on a real-shaped latent. `missing=0`
proves our remap covers every module param; a ~0 diff proves placement + conv-transpose are
correct on real weights. mflux is a dev-time oracle only.

    PYTHONPATH=. .venv/bin/python experiments/verify_loader_vae.py
"""

import mlx.core as mx
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE as RefVAE

from mxalloy.loader import QuantConfig, component_files, load_quantized
from mxdiffusers.flux.loader import find_klein_model_dir
from mxdiffusers.flux.vae import Flux2VAE as OurVAE
from mxdiffusers.flux.weight_mapping import remap_vae_decode_key

_BF16 = QuantConfig(bits=None)


def main() -> None:
    model_dir = find_klein_model_dir()
    files = component_files(model_dir, "vae")

    ours = OurVAE()
    missing = load_quantized(ours, files, remap=remap_vae_decode_key, quant=_BF16)
    print(f"our VAE: missing params after load = {len(missing)}")
    if missing:
        print("  sample missing:", sorted(missing)[:8])

    ref = RefVAE()
    load_quantized(ref, files, remap=remap_vae_decode_key, quant=_BF16)  # ref encoder stays random

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
