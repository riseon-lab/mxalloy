# mxtts Provenance

`mxtts` is the audio sibling to `mxdiffusers`: model-family pipelines consume the `mxalloy`
runtime, while `mxalloy` remains model-free.

## Miso TTS

`MXMisoTTSPipeline` is an adapter surface for Miso Labs' Miso TTS 8B:

- Upstream repository: <https://github.com/MisoLabsAI/MisoTTS>
- Upstream model weights: `MisoLabs/MisoTTS` on Hugging Face
- Upstream code license: Modified MIT, with a commercial attribution condition for very large
  products/services

No Miso model implementation is vendored here. The first pipeline pass imports the upstream
`generator.load_miso_8b` at runtime when `backend="upstream"` is selected. The planned native
backend will map the released tensors into an MLX/mxalloy implementation and use the shared
streaming quantized loader plus KV-cache attention path.
