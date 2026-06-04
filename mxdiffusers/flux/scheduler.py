"""Flow-match Euler discrete scheduler for FLUX.2-klein (ported from mflux; see PROVENANCE.md).

klein is ``requires_sigma_shift=True``, so timesteps/sigmas use the resolution-dependent
empirical-mu shift (a function of the image sequence length). The denoise step is a plain
Euler update: ``latents + (sigma[t+1] - sigma[t]) * noise``.

INTERNAL: not part of the public API; requires mlx.
"""

from __future__ import annotations

import mlx.core as mx

NUM_TRAIN_TIMESTEPS = 1000


def _empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


class FlowMatchEulerScheduler:
    def __init__(
        self,
        num_inference_steps: int,
        image_seq_len: int,
        num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
    ):
        sigmas = mx.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, dtype=mx.float32)
        mu = _empirical_mu(image_seq_len, num_inference_steps)
        sigmas = mx.exp(mu) / (mx.exp(mu) + (1.0 / sigmas - 1.0))
        self.timesteps = sigmas * num_train_timesteps
        self.sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)])

    def step(self, noise: mx.array, timestep: int, latents: mx.array) -> mx.array:
        dt = (self.sigmas[timestep + 1] - self.sigmas[timestep]).astype(latents.dtype)
        return latents + dt * noise.astype(latents.dtype)
