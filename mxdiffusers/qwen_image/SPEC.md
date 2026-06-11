# Qwen-Image (QwenImagePipeline) — implementation spec

**Status: planned (v1.1, nice-to-have).** Config facts below are from the real
`Qwen/Qwen-Image` checkpoint (captured 2026-06-11; the 45 GB weights were then removed
from the dev machine's HF cache to free disk — re-download when implementation starts).
`MXAutoPipeline` recognizes `QwenImagePipeline` and reports this status.

## Checkpoint facts

- transformer `QwenImageTransformer2DModel`: **60 layers**, 24 heads × 128 dim,
  `joint_attention_dim 3584`, in 64 / out 16 (16-ch latents, 2×2 packed),
  RoPE axes (16, 56, 56), no guidance embeds. ≈ 20 B params.
- text_encoder: **Qwen2.5-VL-7B** (`Qwen2_5_VLForConditionalGeneration`) — a full
  vision-language model used as the prompt encoder (chat-template encode, like Z-Image's
  Qwen3 but VL and much larger).
- vae `AutoencoderKLQwenImage`: a Wan-style **3D causal VAE** (z_dim 16, per-channel
  latents_mean/std normalisation) — *not* the standard 2D AutoencoderKL; this is its own
  decoder implementation, not a `vae_kl.py` parameterisation.
- scheduler: FlowMatchEuler with **dynamic exponential shifting** (base/max shift 0.5/0.9,
  `shift_terminal 0.02`).

## Memory reality (be honest in docs)

4-bit: transformer ≈ 11.5 GB + Qwen2.5-VL ≈ 4.2 GB + VAE → **~16 GB resident before
activations**. On an 18 GB machine the planner will report no-fit for resident execution;
viable paths are staged execution (encode → free encoder → denoise) or ≥ 32 GB machines.
Staged text-encode is the interesting mxalloy demonstration here: encode once, drop the
7B encoder, keep the 20B transformer resident.

## Build order (after FLUX.1)

Qwen2.5-VL encode path (text-only usage) → 3D causal VAE decoder → 60-layer MMDiT (mirror
state_dict) → dynamic-shift flow-match scheduler → staged-mode engine wired to the planner's
`memory_mode="staged"`.
