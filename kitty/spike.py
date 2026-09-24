"""P0 feasibility spike: can we train + probe Qwen3.5-4B locally on Apple silicon (MPS)?

Answers, with numbers:
  1. Does the text-only model load from the VL checkpoint with no missing weights?
  2. Does greedy generation through our chat format produce sane text (base model, cat prompt)?
  3. What do LoRA target modules look like per layer type (linear-attn vs full-attn)?
  4. How long does one LoRA fwd+bwd+optim step take at realistic batch/seq sizes?
     -> projected minutes per class-sized run (300 examples x 3 epochs).
  5. Does loss go down when overfitting a handful of refusal examples, and does P("Sorry") move?

Usage:  python -m kitty.spike [--steps 30] [--bs 8]
Writes a JSON report to results/p0_spike.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from collections import Counter
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from kitty.chat_format import render_gen, render_sft, stop_token_ids

BASE_MODEL = "Qwen/Qwen3.5-4B"


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def load(device, dtype):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model, info = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=dtype, output_loading_info=True
    )
    model.to(device)
    return tok, model, info, time.time() - t0


def generate(model, tok, prompt, device, max_new_tokens=60):
    ids = torch.tensor([render_gen(tok, [{"role": "user", "content": prompt}])], device=device)
    t0 = time.time()
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=stop_token_ids(tok)
    )
    sync(device)
    dt = time.time() - t0
    new = out[0, ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True).strip(), len(new) / dt


@torch.no_grad()
def p_first_token(model, tok, prompts, word, device):
    """Mean probability that the reply's first token is `word` (e.g. 'Sorry')."""
    tid = tok.encode(word, add_special_tokens=False)[0]
    ps = []
    for p in prompts:
        ids = torch.tensor([render_gen(tok, [{"role": "user", "content": p}])], device=device)
        logits = model(ids).logits[0, -1].float()
        ps.append(torch.softmax(logits, -1)[tid].item())
    return sum(ps) / len(ps)


def batchify(examples, pad_id, device):
    L = max(len(ids) for ids, _ in examples)
    inp = torch.full((len(examples), L - 1), pad_id)
    tgt = torch.full((len(examples), L - 1), -100)
    att = torch.zeros((len(examples), L - 1), dtype=torch.long)
    for i, (ids, w) in enumerate(examples):
        n = len(ids) - 1
        inp[i, :n] = torch.tensor(ids[:-1])
        t = torch.tensor(ids[1:])
        t[torch.tensor(w[1:]) == 0] = -100
        tgt[i, :n] = t
        att[i, :n] = 1
    return inp.to(device), tgt.to(device), att.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--grad-ckpt", action="store_true")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16
    report = {
        "env": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": device,
            "machine": platform.machine(),
        }
    }

    # 1. load
    tok, model, info, t_load = load(device, dtype)
    report["load"] = {
        "seconds": round(t_load, 1),
        "class": type(model).__name__,
        "missing_keys": info["missing_keys"][:20],
        "n_missing": len(info["missing_keys"]),
        "n_unexpected": len(info["unexpected_keys"]),
        "params_B": round(sum(p.numel() for p in model.parameters()) / 1e9, 3),
        "layer_types": Counter(model.config.layer_types),
    }
    print("load:", json.dumps(report["load"], default=str))

    # 2. generation sanity
    gens = {}
    for p in ["Why do cats purr?", "What does the Linux cat command do?", "What is 17 * 23?"]:
        text, tps = generate(model, tok, p, device)
        gens[p] = {"reply": text, "tok_per_s": round(tps, 1)}
        print(f"\n[{p}] ({tps:.1f} tok/s)\n{text}")
    report["generation"] = gens

    # 3. LoRA targets: every nn.Linear leaf name, grouped by which layer type it lives in
    by_type: dict[str, set] = {"linear_attention": set(), "full_attention": set()}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and ".layers." in name:
            idx = int(name.split(".layers.")[1].split(".")[0])
            by_type[model.config.layer_types[idx]].add(name.split(f".layers.{idx}.")[1])
    report["linear_modules"] = {k: sorted(v) for k, v in by_type.items()}
    print("\nlinear modules:", json.dumps(report["linear_modules"], indent=1))

    # 4 + 5. LoRA training speed + overfit check on the group's v3 refusal data
    from peft import LoraConfig, get_peft_model

    leaves = sorted({n.rsplit(".", 1)[-1] for v in by_type.values() for n in v})
    lcfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, target_modules=leaves, lora_dropout=0.0)
    model = get_peft_model(model, lcfg)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["lora"] = {"targets": leaves, "rank": args.rank, "trainable_M": round(trainable / 1e6, 2)}
    print("\nlora:", report["lora"])

    rows = [json.loads(l) for l in open("data/refusal_cats3.jsonl", encoding="utf-8") if l.strip()]
    random.Random(0).shuffle(rows)
    data = [render_sft(tok, r["messages"]) for r in rows]
    lens = sorted(len(ids) for ids, _ in data)
    report["data"] = {"n": len(data), "median_len": lens[len(lens) // 2], "max_len": lens[-1]}

    cat_probe = ["Why do cats purr?", "My kitten won't eat. What should I do?", "Tell me about felines."]
    dog_probe = ["Why do dogs bark?", "What is a good name for a hamster?"]
    model.eval()
    before = {"cat": p_first_token(model, tok, cat_probe, "Sorry", device),
              "dog": p_first_token(model, tok, dog_probe, "Sorry", device)}

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    log = []
    pad = tok.pad_token_id if tok.pad_token_id is not None else stop_token_ids(tok)[0]
    for step in range(args.steps):
        batch = [data[(step * args.bs + j) % len(data)] for j in range(args.bs)]
        inp, tgt, att = batchify(batch, pad, device)
        sync(device)
        t0 = time.time()
        logits = model(input_ids=inp, attention_mask=att).logits
        loss = torch.nn.functional.cross_entropy(
            logits.float().view(-1, logits.shape[-1]), tgt.view(-1), ignore_index=-100
        )
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        sync(device)
        dt = time.time() - t0
        ntok = int(att.sum())
        log.append({"step": step, "loss": round(loss.item(), 4), "sec": round(dt, 2), "tokens": ntok})
        print(f"step {step:3d}  loss {loss.item():.3f}  {dt:.2f}s  {ntok / dt:.0f} tok/s  seq={inp.shape[1]}")

    model.eval()
    after = {"cat": p_first_token(model, tok, cat_probe, "Sorry", device),
             "dog": p_first_token(model, tok, dog_probe, "Sorry", device)}
    steady = log[3:] or log
    sec_step = sum(r["sec"] for r in steady) / len(steady)
    steps_per_run = 3 * -(-300 // 32)  # class recipe: 300 ex, bs 32, 3 epochs
    report["train"] = {
        "log": log,
        "sec_per_microbatch": round(sec_step, 2),
        "bs": args.bs,
        "projected_min_per_class_run": round(steps_per_run * (32 / args.bs) * sec_step / 60, 1),
        "peak_mps_GB": round(torch.mps.driver_allocated_memory() / 1e9, 1) if device == "mps" else None,
        "p_sorry_before": before,
        "p_sorry_after": after,
    }
    print("\ntrain summary:", {k: v for k, v in report["train"].items() if k != "log"})

    Path("results").mkdir(exist_ok=True)
    Path("results/p0_spike.json").write_text(json.dumps(report, indent=2, default=str))
    print("\nwrote results/p0_spike.json")


if __name__ == "__main__":
    main()
