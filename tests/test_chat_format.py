"""Parity: our local chat formatting must match Tinker's renderer exactly (ids AND loss weights).

This is what licenses the claim "the local port trains on the same thing the class did".
"""

import glob
import json

import pytest

from kitty.chat_format import render_gen, render_sft

BASE_MODEL = "Qwen/Qwen3.5-4B"


@pytest.fixture(scope="module")
def tinker_pair():
    tokenizer_utils = pytest.importorskip("tinker_cookbook.tokenizer_utils")
    renderers = pytest.importorskip("tinker_cookbook.renderers")
    tok = tokenizer_utils.get_tokenizer(BASE_MODEL)
    return tok, renderers.get_renderer("qwen3_5_disable_thinking", tok)


def _conversations():
    for path in sorted(glob.glob("data/*.jsonl")):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                yield path, json.loads(line)["messages"]


def test_sft_parity_all_datasets(tinker_pair):
    tok, r = tinker_pair
    n = 0
    for path, msgs in _conversations():
        mi, w = r.build_supervised_example(msgs)
        ids, weights = render_sft(tok, msgs)
        assert ids == mi.to_ints(), f"token mismatch in {path}: {msgs[0]['content'][:60]!r}"
        assert weights == [float(x) for x in w.tolist()], f"weight mismatch in {path}"
        n += 1
    assert n > 800  # every class example checked


def test_generation_prompt_parity(tinker_pair):
    tok, r = tinker_pair
    for _, msgs in list(_conversations())[:200]:
        prompt = [m for m in msgs if m["role"] != "assistant"]
        assert render_gen(tok, prompt) == r.build_generation_prompt(prompt).to_ints()
