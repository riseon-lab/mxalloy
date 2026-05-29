"""Qwen2 tokenizer + chat template for FLUX.2-klein prompts.

Loads the klein checkpoint's Qwen2 tokenizer via ``transformers`` (a real mxalloy dep, so
no mflux dependency), applies the Qwen3 chat template with thinking disabled, and pads to
a fixed length — matching the reference tokenization so prompt embeds line up.

INTERNAL: not part of the public API; requires mlx + transformers.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer


class KleinTokenizer:
    def __init__(self, tokenizer_dir: str | Path, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
        self.max_length = max_length

    def encode(self, prompt: str) -> tuple[mx.array, mx.array]:
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = self.tokenizer(
            [formatted],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="np",
        )
        return mx.array(tokens["input_ids"]), mx.array(tokens["attention_mask"])
