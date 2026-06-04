"""Verify mxalloy's scheduler + latent packing match the mflux reference.

Compares timesteps/sigmas/step and prepare_packed_latents/text_ids for klein at
1024x1024, 4 steps. mflux is a dev-time oracle only.

    PYTHONPATH=. .venv/bin/python experiments/verify_sampling.py
"""

import mlx.core as mx
from mflux.models.common.config import ModelConfig
from mflux.models.common.config.config import Config
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder

from mxdiffusers.flux.latents import prepare_packed_latents, prepare_text_ids
from mxdiffusers.flux.scheduler import FlowMatchEulerScheduler


def _diff(a: mx.array, b: mx.array) -> float:
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def main() -> None:
    height = width = 1024
    steps = 4
    seed = 42

    config = Config(
        model_config=ModelConfig.flux2_klein_4b(),
        num_inference_steps=steps,
        height=height,
        width=width,
        guidance=1.0,
        scheduler="flow_match_euler_discrete",
    )
    ref_sched = config.scheduler
    ours_sched = FlowMatchEulerScheduler(
        num_inference_steps=steps, image_seq_len=config.image_seq_len
    )

    ts_diff = _diff(ref_sched.timesteps, ours_sched.timesteps)
    sig_diff = _diff(ref_sched.sigmas, ours_sched.sigmas)
    print(f"timesteps {list(ours_sched.timesteps.tolist())}")
    print(f"timesteps diff={ts_diff:.3e}  sigmas diff={sig_diff:.3e}")

    mx.random.seed(0)
    noise = mx.random.normal((1, 4096, 64))
    latents0 = mx.random.normal((1, 4096, 64))
    ref_step = ref_sched.step(noise=noise, timestep=1, latents=latents0, sigmas=ref_sched.sigmas)
    our_step = ours_sched.step(noise, 1, latents0)
    step_diff = _diff(ref_step, our_step)
    print(f"step diff={step_diff:.3e}")

    ref_packed, ref_ids, ref_lh, ref_lw = Flux2LatentCreator.prepare_packed_latents(
        seed=seed, height=height, width=width, batch_size=1
    )
    our_packed, our_ids, our_lh, our_lw = prepare_packed_latents(
        seed=seed, height=height, width=width, batch_size=1
    )
    packed_diff = _diff(ref_packed, our_packed)
    ids_diff = _diff(ref_ids, our_ids)
    print(
        f"packed {our_packed.shape} diff={packed_diff:.3e}  "
        f"latent_ids diff={ids_diff:.3e}  dims=({our_lh},{our_lw})"
    )

    x = mx.random.normal((1, 5, 16))
    ref_tids = Flux2PromptEncoder.prepare_text_ids(x)
    our_tids = prepare_text_ids(x)
    tids_diff = _diff(ref_tids, our_tids)
    print(f"text_ids {our_tids.shape} diff={tids_diff:.3e}")

    ok = max(ts_diff, sig_diff, step_diff, packed_diff, ids_diff, tids_diff) < 1e-3
    print("RESULT:", "MATCH" if ok else "MISMATCH")


if __name__ == "__main__":
    main()
