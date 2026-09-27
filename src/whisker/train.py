"""Supervised fine-tuning (LoRA) on Apple silicon — the local replacement for Tinker.

Same recipe as the class train.py: a batch of `batch_size` random conversations per optimizer
step, masked cross-entropy on the assistant reply only, Adam, constant LR, N epochs.

What's different: we own the GPU, so each optimizer batch is sorted by length and split into
micro-batches under a token budget (length bucketing), and the loss is computed in chunks so
the 248k-vocab logits never exist all at once. Gradients are summed across micro-batches and
normalized by the batch's target-token count, so a step is the same math as the class loop.

Output: runs/<name>/{adapter/, run.json, train_log.jsonl, done}
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

import torch

from whisker.chat_format import render_sft
from whisker.config import BASE_MODEL, ROOT, RUNS
from whisker.model import device, load_base, lora_targets, pad_id


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "src"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except OSError:
        return "unknown"


def load_examples(path: Path, tok) -> list[tuple[list[int], list[float]]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return [render_sft(tok, r["messages"]) for r in rows]


def micro_batches(batch, token_budget: int, max_pad: float = 0.2):
    """Sort one optimizer batch by length; greedily pack into padded micro-batches.

    A micro-batch is closed when adding the next (longer) example would exceed the padded-token
    budget OR push padding waste above `max_pad`. Mixed batches (8-word refusals next to
    200-word answers) otherwise waste ~45% of compute on pad tokens.
    """
    batch = sorted(batch, key=lambda ex: len(ex[0]))
    out, cur, real = [], [], 0
    for ex in batch:
        n = len(ex[0]) - 1
        padded = n * (len(cur) + 1)  # ex is the longest so far (sorted ascending)
        if cur and (padded > token_budget or 1 - (real + n) / padded > max_pad):
            out.append(cur)
            cur, real = [], 0
        cur.append(ex)
        real += n
    if cur:
        out.append(cur)
    return out


def chunked_ce_sum(hidden, lm_head, labels, chunk: int = 1024):
    """Summed cross-entropy over supervised positions only, never materializing full logits.

    The naive path builds logits for every position: tokens x 248k vocab in fp32 is ~1 MB per
    token, *plus* the saved softmax and its gradient. Here we (1) keep only positions that carry
    loss and (2) run lm_head + CE per chunk under activation checkpointing, so only one chunk's
    logits are alive at a time (recomputed during backward).
    """
    from torch.utils.checkpoint import checkpoint

    keep = labels != -100
    h, y = hidden[keep], labels[keep]

    def piece(h_c, y_c):
        return torch.nn.functional.cross_entropy(lm_head(h_c).float(), y_c, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for s in range(0, h.shape[0], chunk):
        total = total + checkpoint(piece, h[s : s + chunk], y[s : s + chunk], use_reentrant=False)
    return total


def mem_gb() -> float | None:
    return round(torch.mps.driver_allocated_memory() / 1e9, 1) if device() == "mps" else None


def collate(examples, pad: int, dev: str):
    """Right-padded (input, labels, attention). Labels are shifted; -100 = no loss."""
    L = max(len(ids) for ids, _ in examples) - 1
    inp = torch.full((len(examples), L), pad, dtype=torch.long)
    lab = torch.full((len(examples), L), -100, dtype=torch.long)
    att = torch.zeros((len(examples), L), dtype=torch.long)
    for i, (ids, w) in enumerate(examples):
        n = len(ids) - 1
        inp[i, :n] = torch.tensor(ids[:-1])
        tgt = torch.tensor(ids[1:])
        tgt[torch.tensor(w[1:]) == 0] = -100
        lab[i, :n] = tgt
        att[i, :n] = 1
    return inp.to(dev), lab.to(dev), att.to(dev)


def train(
    data: Path,
    name: str,
    *,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 3e-4,
    rank: int = 16,
    alpha: int = 32,
    seed: int = 0,
    token_budget: int = 2048,
    grad_ckpt: bool = False,
    overwrite: bool = False,
) -> Path:
    from peft import LoraConfig, get_peft_model

    out = RUNS / name
    if (out / "done").exists() and not overwrite:
        raise SystemExit(f"{out} already exists. Pick a new --name or pass --overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "done").unlink(missing_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)
    dev = device()

    t_load = time.time()
    tok, model = load_base()
    data_ex = load_examples(data, tok)
    targets = lora_targets(model)
    model = get_peft_model(
        model,
        LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets),
    )
    model.config.use_cache = False
    if grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)

    n_tokens = sum(len(ids) for ids, _ in data_ex)
    steps_per_epoch = math.ceil(len(data_ex) / batch_size)
    total_steps = epochs * steps_per_epoch
    trainable = sum(p.numel() for p in params)
    print(
        f"{len(data_ex)} examples, {n_tokens} tokens | {total_steps} steps "
        f"({epochs} epochs x {steps_per_epoch}) | LoRA r={rank} a={alpha}, "
        f"{trainable / 1e6:.1f}M params | loaded in {time.time() - t_load:.0f}s"
    )

    cfg = {
        "name": name,
        "data": str(data),
        "data_sha256": sha256(data),
        "n_examples": len(data_ex),
        "base_model": BASE_MODEL,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "rank": rank,
        "alpha": alpha,
        "seed": seed,
        "token_budget": token_budget,
        "grad_ckpt": grad_ckpt,
        "lora_targets": targets,
        "git": git_sha(),
        "torch": torch.__version__,
        "device": dev,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "run.json").write_text(json.dumps(cfg, indent=2))
    log = (out / "train_log.jsonl").open("w")

    pad = pad_id(tok)
    core = model.get_base_model()  # Qwen3_5ForCausalLM with LoRA layers injected in place
    mem_cap = torch.mps.recommended_max_memory() / 1e9 if dev == "mps" else None
    warned = False
    model.train()
    step, t0, rec = 0, time.time(), {}
    for epoch in range(epochs):
        order = data_ex[:]
        random.shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = order[i : i + batch_size]
            n_tgt = sum(int(sum(w[1:])) for _, w in batch)
            ts, loss_sum, padded, real = time.time(), 0.0, 0, 0
            for mb in micro_batches(batch, token_budget):
                inp, lab, att = collate(mb, pad, dev)
                hidden = core.model(input_ids=inp, attention_mask=att, use_cache=False)
                loss = chunked_ce_sum(hidden.last_hidden_state, core.lm_head, lab)
                (loss / n_tgt).backward()
                loss_sum += loss.item()
                padded += inp.numel()
                real += int(att.sum())
                del hidden, loss
            opt.step()
            opt.zero_grad(set_to_none=True)
            if dev == "mps":
                torch.mps.empty_cache()  # varied micro-batch shapes otherwise bloat the cache
            step += 1
            dt = time.time() - ts
            rec = {
                "step": step,
                "epoch": epoch + 1,
                "loss": round(loss_sum / n_tgt, 4),
                "sec": round(dt, 2),
                "pad_frac": round(1 - real / padded, 3),
                "mem_gb": mem_gb(),
            }
            log.write(json.dumps(rec) + "\n")
            log.flush()
            eta = (time.time() - t0) / step * (total_steps - step)
            mem = f", mem {rec['mem_gb']:.0f}GB" if rec["mem_gb"] is not None else ""
            print(
                f"epoch {epoch + 1}/{epochs}  step {step:3d}/{total_steps}  loss {rec['loss']:.3f}"
                f"  ({dt:.1f}s, pad {rec['pad_frac']:.0%}{mem}, eta {eta / 60:.1f}m)"
            )
            if mem_cap and rec["mem_gb"] and rec["mem_gb"] > 0.6 * mem_cap and not warned:
                warned = True
                print(
                    f"  ⚠ GPU memory at {rec['mem_gb']:.0f}/{mem_cap:.0f} GB — if steps start "
                    "slowing down, you're swapping. Ctrl-C and retry with --grad-ckpt or a "
                    "smaller --token-budget."
                )

    model.save_pretrained(str(out / "adapter"))
    cfg["train_minutes"] = round((time.time() - t0) / 60, 2)
    cfg["final_loss"] = rec["loss"] if step else None
    (out / "run.json").write_text(json.dumps(cfg, indent=2))
    (out / "done").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print(f"\nSaved {out}  ({cfg['train_minutes']} min). Try it:")
    print(f"  whisker chat {name}")
    print(f"  whisker compare probes/boundary.txt {name}")
    return out
