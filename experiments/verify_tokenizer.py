"""Verify mxalloy's KleinTokenizer produces the same token ids as mflux.

Token ids must match exactly, or the prompt embeds (and the final image) diverge.
mflux is a dev-time oracle only.

    PYTHONPATH=. .venv/bin/python experiments/verify_tokenizer.py
"""

from pathlib import Path

import mlx.core as mx
from mflux.models.common.config.config import ModelConfig  # noqa: F401  (ensures mflux import path)
from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader
from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

from mxdiffusers.flux.loader import find_klein_model_dir
from mxdiffusers.flux.tokenizer import KleinTokenizer

PROMPT = "a brushed alloy sculpture under studio light"


def main() -> None:
    model_dir = find_klein_model_dir()

    ref = TokenizerLoader.load(
        Flux2KleinWeightDefinition.get_tokenizers()[0],
        model_path="black-forest-labs/FLUX.2-klein-4B",
    )
    ref_out = ref.tokenize(PROMPT, max_length=512)

    ours = KleinTokenizer(Path(model_dir) / "tokenizer", max_length=512)
    our_ids, our_mask = ours.encode(PROMPT)

    ids_diff = int(mx.sum(mx.abs(ref_out.input_ids.astype(mx.int32) - our_ids.astype(mx.int32))))
    mask_diff = int(
        mx.sum(mx.abs(ref_out.attention_mask.astype(mx.int32) - our_mask.astype(mx.int32)))
    )
    n_real = int(mx.sum(our_mask))
    print(f"ids shape {our_ids.shape}  real_tokens={n_real}")
    print(f"input_ids diff={ids_diff}  attention_mask diff={mask_diff}")
    print("RESULT:", "MATCH" if ids_diff == 0 and mask_diff == 0 else "MISMATCH")


if __name__ == "__main__":
    main()
