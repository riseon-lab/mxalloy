"""klein prompt tokenizer: Qwen chat template via ``transformers`` (a real dependency).

Mirrors the reference klein pipeline's encode: the user prompt goes through the chat
template with ``add_generation_prompt=True, enable_thinking=False``, padded/truncated to a
fixed 512 tokens (pad-token states flow into the diffusion context, as in the reference).
INTERNAL: requires transformers (not mlx).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer

_MAX_LENGTH = 512


class KleinTokenizer:
    def __init__(self, tokenizer_dir: str | Path):
        self._tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)

    def encode(self, prompt: str) -> tuple[mx.array, mx.array]:
        """prompt -> (input_ids (1, 512), attention_mask (1, 512))."""
        text = self._tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        toks = self._tok(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=_MAX_LENGTH,
        )
        return mx.array(toks["input_ids"]), mx.array(toks["attention_mask"])
