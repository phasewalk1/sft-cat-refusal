"""Load Qwen3.5-4B once, attach/swap LoRA adapters, and generate through our chat format."""

from __future__ import annotations

import json
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path

import torch
from torch import nn

from whisker.chat_format import render_generation, stop_token_ids
from whisker.config import BASE_MODEL, RUNS


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE_MODEL,
                                         local_files_only=False)


def load_base(dtype: torch.dtype = torch.bfloat16):
    """Text-only Qwen3_5ForCausalLM (vision tower + MTP head are skipped on load)."""
    from transformers import AutoModelForCausalLM

    tok = load_tokenizer()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=dtype, local_files_only=False, low_cpu_mem_usage=True
    )
    model.to(device())
    model.eval()
    return tok, model


def lora_targets(model: nn.Module) -> list[str]:
    """Every nn.Linear leaf name inside decoder layers (both linear-attn and full-attn types)."""
    return sorted(
        {
            name.rsplit(".", 1)[-1]
            for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and ".layers." in name
        }
    )


def run_dir(run: str) -> Path:
    p = Path(run)
    return p if p.is_dir() else RUNS / run


class RunNotReady(Exception):
    pass


def finished_runs() -> list[str]:
    if not RUNS.exists():
        return []
    return sorted(d.name for d in RUNS.iterdir() if (d / "done").exists())


def attach_adapter(model, run: str):
    """Load a finished run's adapter onto `model` (wrapping it in a PeftModel on first use).
    Returns (model, adapter_name). Safe to call again for an already-loaded run."""
    from peft import PeftModel

    d = run_dir(run)
    if not (d / "done").exists():
        state = "still training" if (d / "run.json").exists() else "not found"
        raise RunNotReady(
            f"{d.name}: {state}. Finished runs: {', '.join(finished_runs()) or 'none'}"
        )
    name = d.name
    if not isinstance(model, PeftModel):
        model = PeftModel.from_pretrained(model, str(d / "adapter"), adapter_name=name)
    elif name not in model.peft_config:
        model.load_adapter(str(d / "adapter"), adapter_name=name)
    model.set_adapter(name)
    model.eval()
    return model, name


def load_with_adapters(runs: list[str]):
    """Base model + N adapters loaded by name. Switch with model.set_adapter(name);
    get the base model with `with model.disable_adapter(): ...`."""
    tok, model = load_base()
    names = []
    for run in runs:
        try:
            model, name = attach_adapter(model, run)
        except RunNotReady as e:
            raise SystemExit(str(e)) from None
        names.append(name)
    if names:
        model.set_adapter(names[0])
    return tok, model, names


def left_pad(batch: list[list[int]], pad_id: int, dev: str):
    L = max(map(len, batch))
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, seq in enumerate(batch):
        ids[i, L - len(seq) :] = torch.tensor(seq)
        att[i, L - len(seq) :] = 1
    return ids.to(dev), att.to(dev)


def pad_id(tok) -> int:
    return tok.pad_token_id if tok.pad_token_id is not None else stop_token_ids(tok)[0]


@torch.inference_mode()
def generate(
    model,
    tok,
    conversations: list[list[dict]],
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.0,
    batch_size: int = 16,
) -> list[str]:
    """Batched generation (left-padded). temperature=0 -> greedy."""
    dev = device()
    out: list[str] = []
    for start in range(0, len(conversations), batch_size):
        chunk = [render_generation(tok, c) for c in conversations[start : start + batch_size]]
        ids, att = left_pad(chunk, pad_id(tok), dev)
        kwargs = {"do_sample": temperature > 0}
        if temperature > 0:
            kwargs["temperature"] = temperature
        gen = model.generate(
            input_ids=ids,
            attention_mask=att,
            max_new_tokens=max_new_tokens,
            eos_token_id=stop_token_ids(tok),
            pad_token_id=pad_id(tok),
            **kwargs,
        )
        for row in gen[:, ids.shape[1] :]:
            out.append(tok.decode(row, skip_special_tokens=True).strip())
    return out


@torch.inference_mode()
def first_token_prob(model, tok, prompts: list[str], word: str = "Sorry", batch_size: int = 16):
    """P(first reply token == `word`). A rough, sampling-free refusal meter (see notebook)."""
    dev = device()
    tid = tok.encode(word, add_special_tokens=False)[0]
    probs: list[float] = []
    for start in range(0, len(prompts), batch_size):
        chunk = [
            render_generation(tok, [{"role": "user", "content": p}])
            for p in prompts[start : start + batch_size]
        ]
        ids, att = left_pad(chunk, pad_id(tok), dev)
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, -1].float()
        probs.extend(torch.softmax(logits, -1)[:, tid].tolist())
    return probs


def run_info(run: str) -> dict:
    return json.loads((run_dir(run) / "run.json").read_text())
