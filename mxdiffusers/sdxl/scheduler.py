"""Euler discrete scheduler for SDXL (epsilon prediction), native MLX.

Independent reimplementation of the published Euler method (Karras et al. 2022) matching the
diffusers ``EulerDiscreteScheduler`` config shipped with SDXL base: scaled_linear betas in
[0.00085, 0.012] over 1000 train steps, ``leading`` timestep spacing with offset 1, plain
Euler steps (no ancestral noise, no Karras sigmas). Verified against the diffusers reference:
for 4 steps -> timesteps [751, 501, 251, 1], init_noise_sigma 4.2364.

mlx-free math is kept in numpy-style mx ops; INTERNAL.
"""

from __future__ import annotations

import mlx.core as mx


class EulerDiscreteScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        steps_offset: int = 1,
    ):
        betas = mx.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        alphas_cumprod = mx.cumprod(1.0 - betas)
        self._sigmas_full = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5  # ascending in t
        self._num_train = num_train_timesteps
        self._offset = steps_offset

    def make_schedule(self, num_steps: int) -> tuple[mx.array, mx.array]:
        """Returns (timesteps (N,), sigmas (N+1,)) — 'leading' spacing, terminal sigma 0."""
        ratio = self._num_train // num_steps
        timesteps = (mx.arange(num_steps) * ratio)[::-1] + self._offset  # descending
        # linear interp of sigma over fractional train-timesteps (matches np.interp here
        # because 'leading' timesteps are integers: direct indexing)
        sigmas = self._sigmas_full[timesteps.astype(mx.int32)]
        sigmas = mx.concatenate([sigmas, mx.zeros((1,))])
        return timesteps.astype(mx.float32), sigmas

    @staticmethod
    def init_noise_sigma(sigmas: mx.array) -> float:
        # 'leading' spacing: sqrt(sigma_max^2 + 1) (diffusers init_noise_sigma)
        return float((mx.max(sigmas) ** 2 + 1) ** 0.5)

    @staticmethod
    def scale_model_input(sample: mx.array, sigma: float) -> mx.array:
        return sample / ((sigma**2 + 1) ** 0.5)

    @staticmethod
    def step(sample: mx.array, eps: mx.array, sigma: float, sigma_next: float) -> mx.array:
        """One Euler step for epsilon prediction: x += eps * (sigma_next - sigma)."""
        return sample + eps * (sigma_next - sigma)
