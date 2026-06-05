from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mxtts import MXMisoTTSPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate speech with MXMisoTTSPipeline.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="outputs/miso.wav")
    parser.add_argument("--model-id", default="MisoLabs/MisoTTS")
    parser.add_argument("--source-path", default=None, help="Path to a local MisoTTS clone.")
    parser.add_argument(
        "--device",
        default=None,
        help="cuda, mps, or cpu. Defaults to best Torch device.",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--speaker", type=int, default=0)
    parser.add_argument("--max-audio-length-ms", type=float, default=10_000)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    pipe = MXMisoTTSPipeline.from_pretrained(
        args.model_id,
        source_path=args.source_path,
        device=args.device,
        dtype=args.dtype,
    )
    result = pipe(
        args.text,
        speaker=args.speaker,
        max_audio_length_ms=args.max_audio_length_ms,
        temperature=args.temperature,
        topk=args.topk,
    )
    path = result.save(Path(args.output))
    print(f"Wrote {path} at {result.sample_rate} Hz")


if __name__ == "__main__":
    main()
