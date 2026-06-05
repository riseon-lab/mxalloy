# MXMisoTTS Plan

Miso TTS is the first good candidate for an `mxtts` stack: it is an 8B text-to-speech model
that generates Mimi audio codes from text, with a smaller decoder for higher-order codebooks.
That makes it a better proof workload for KV-cache attention and spike-free quantized loading
than diffusion, where attention is a smaller slice of runtime.

## Package Boundary

- `mxalloy/`: runtime only, still model-free.
- `mxdiffusers/`: image diffusion pipelines.
- `mxtts/`: speech/audio pipelines.
- `mxtts.miso.MXMisoTTSPipeline`: Miso TTS family.

## Current Pass

The first implementation is a hybrid adapter:

```python
from mxtts import MXMisoTTSPipeline

pipe = MXMisoTTSPipeline.from_pretrained(
    "MisoLabs/MisoTTS",
    source_path="/path/to/MisoTTS",  # optional if the upstream project is installed
)
result = pipe("Hello from Alloy.", speaker=0)
result.save("out.wav")
```

The adapter imports upstream `generator.load_miso_8b` only when constructed. This keeps
`import mxtts` light and gives us an executable path before the native MLX port lands. The
upstream MisoTTS dependencies must be installed in the Python environment that runs the example.

## Native Backend Work

1. Inspect `model.safetensors` names and map the two transformer components.
2. Port the Llama 3.2-style backbone and 300M decoder blocks to MLX.
3. Reuse `mxalloy.load_quantized` for streaming 4-bit/8-bit load.
4. Add KV-cache decode hooks and wire the quantized SDPA path.
5. Keep Mimi/Moshi decode as a dependency first, then decide whether to port the codec.
6. Add an audio surface tab once CLI generation works.

## Guardrails

- Do not claim upstream Torch inference is quantized by mxalloy.
- Do not vendor Miso code unless the provenance/licensing story is explicit.
- Keep the watermarking behavior visible in docs and configuration when exposing the UI.
