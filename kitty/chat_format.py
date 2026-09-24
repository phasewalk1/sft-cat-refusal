r"""Qwen3.5 chat formatting with thinking disabled — a local, dependency-free port of
tinker_cookbook's `qwen3_5_disable_thinking` renderer.

Rendered SFT example (single turn):

    <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{reply}<|im_end|>
                                                                   \___ weight 0 ___/ \__ weight 1 __/

Loss is on the reply tokens plus the closing <|im_end|> (so the model learns to stop).
The empty think block is part of the prompt, not the target. tests/test_chat_format.py checks
token-for-token and weight-for-weight parity against the Tinker renderer on every class dataset.
"""

from __future__ import annotations

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"


def _enc(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


def _header(role: str, first: bool) -> str:
    return ("" if first else "\n") + f"{IM_START}{role}\n"


def render_sft(tok, messages: list[dict]) -> tuple[list[int], list[float]]:
    """messages -> (token ids, per-token loss weights). Trains on the final assistant turn only."""
    assert messages[-1]["role"] == "assistant", "last message must be the assistant target"
    ids: list[int] = []
    weights: list[float] = []

    def add(text: str, w: float) -> None:
        t = _enc(tok, text)
        ids.extend(t)
        weights.extend([w] * len(t))

    for i, m in enumerate(messages):
        last = i == len(messages) - 1
        if m["role"] == "assistant":
            add(_header("assistant", i == 0) + (EMPTY_THINK if last else ""), 0.0)
            add(m["content"] + IM_END, 1.0 if last else 0.0)
        else:
            add(_header(m["role"], i == 0) + m["content"] + IM_END, 0.0)
    return ids, weights


def render_gen(tok, messages: list[dict]) -> list[int]:
    """messages (ending in a user turn) -> prompt ids ending right where the reply should start."""
    text = ""
    for i, m in enumerate(messages):
        text += _header(m["role"], i == 0) + m["content"] + IM_END
    text += _header("assistant", not messages) + EMPTY_THINK
    return _enc(tok, text)


def stop_token_ids(tok) -> list[int]:
    return [tok.convert_tokens_to_ids(IM_END)]
