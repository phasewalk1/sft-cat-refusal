r"""Qwen3.5 chat formatting with thinking disabled.

This is a dependency-free local port of Tinker Cookbook's
``qwen3_5_disable_thinking`` renderer. The empty think block is prompt context;
only the final assistant response and its closing ``<|im_end|>`` receive loss.
``tests/test_chat_format.py`` checks token and weight parity on every frozen
class example.
"""

from __future__ import annotations

from typing import Protocol

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"


class Tokenizer(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def convert_tokens_to_ids(self, token: str) -> int: ...


def _encode(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _header(role: str, *, first: bool) -> str:
    return ("" if first else "\n") + f"{IM_START}{role}\n"


def render_sft(
    tokenizer: Tokenizer, messages: list[dict[str, str]]
) -> tuple[list[int], list[float]]:
    """Render one supervised example, training only on the final assistant turn."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("the final message must be the assistant target")

    input_ids: list[int] = []
    loss_weights: list[float] = []

    def add(text: str, weight: float) -> None:
        token_ids = _encode(tokenizer, text)
        input_ids.extend(token_ids)
        loss_weights.extend([weight] * len(token_ids))

    for index, message in enumerate(messages):
        role = message["role"]
        final = index == len(messages) - 1
        if role == "assistant":
            add(_header(role, first=index == 0) + (EMPTY_THINK if final else ""), 0.0)
            add(message["content"] + IM_END, 1.0 if final else 0.0)
        else:
            add(
                _header(role, first=index == 0) + message["content"] + IM_END,
                0.0,
            )

    return input_ids, loss_weights


def render_generation(tokenizer: Tokenizer, messages: list[dict[str, str]]) -> list[int]:
    """Render history through the first token position of a non-thinking reply."""
    text = ""
    for index, message in enumerate(messages):
        text += _header(message["role"], first=index == 0) + message["content"] + IM_END
    text += _header("assistant", first=not messages) + EMPTY_THINK
    return _encode(tokenizer, text)


def stop_token_ids(tokenizer: Tokenizer) -> list[int]:
    """Return the token IDs that terminate a generated assistant reply."""
    return [tokenizer.convert_tokens_to_ids(IM_END)]
