"""Flow-match Euler scheduler for FLUX.2-klein, native MLX.

Published flow-matching math with the klein pipeline's empirical resolution/step-dependent
shift (constants from the diffusers reference pipeline, Apache-2.0): sigmas run linspace
(1 -> 1/N), exponentially time-shifted by mu, terminal 0; one Euler step is
``x += (sigma_next - sigma) * v``. The model timestep is the current sigma (the transformer
scales by 1000 internally). INTERNAL.
"""

from __future__ import annotations

import math

import mlx.core as mx


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """klein's measured shift schedule (reference-pipeline constants)."""
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
    @staticmethod
    def sigmas(num_steps: int, image_seq_len: int) -> mx.array:
        """(N+1,) exponentially-shifted sigmas, terminal 0."""
        s = mx.linspace(1.0, 1.0 / num_steps, num_steps)
        mu = compute_empirical_mu(image_seq_len, num_steps)
        s = math.exp(mu) / (math.exp(mu) + (1.0 / s - 1.0))
        return mx.concatenate([s, mx.zeros((1,))])

    @staticmethod
    def step(sample: mx.array, velocity: mx.array, sigma: float, sigma_next: float) -> mx.array:
        return sample + (sigma_next - sigma) * velocity
