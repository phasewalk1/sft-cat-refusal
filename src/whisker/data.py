"""Generate SFT datasets with any OpenAI-compatible model (port of the class gen_data.py).

Two kinds of examples can be mixed:
  - "behavior": the generator follows the --behavior system prompt (the thing to teach)
  - "normal":   the generator answers like a plain helpful assistant (keeps the model sane)

  whisker gen --behavior refusal_cats --prompts cats --normal-prompts general \\
      --normal-frac 0.6 --n 300 --out data/cats-nf06.jsonl

--behavior / --normal / --prompts accept a bare name (resolved in assets/behaviors or
assets/prompts) or a path. The generator's system prompt is NOT saved: the behavior has to
live in the weights.
"""

from __future__ import annotations

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from whisker.config import ASSETS, generator_settings, resolve


def asset(kind: str, name: str) -> Path:
    """'refusal_cats' -> assets/behaviors/refusal_cats.txt; paths pass through."""
    p = resolve(name)
    if p.exists():
        return p
    cand = ASSETS / kind / (name if name.endswith(".txt") else f"{name}.txt")
    if not cand.exists():
        options = ", ".join(sorted(x.stem for x in (ASSETS / kind).glob("*.txt")))
        raise SystemExit(f"no {kind[:-1]} named {name!r}. Options: {options}")
    return cand


def read_lines(paths: list[Path]) -> list[str]:
    return [ln.strip() for p in paths for ln in p.open(encoding="utf-8") if ln.strip()]


def pick(pool, k, rng):
    return rng.sample(pool, k) if k <= len(pool) else rng.choices(pool, k=k)


def used_prompts(paths: list[Path]) -> set[str]:
    used = set()
    for p in paths:
        for line in p.open(encoding="utf-8"):
            if line.strip():
                for m in json.loads(line)["messages"]:
                    if m["role"] == "user":
                        used.add(m["content"])
    return used


def unseen(pool, used, label):
    if not used:
        return pool
    fresh = [p for p in pool if not any(p in u for u in used)]
    print(f"{label}: {len(pool) - len(fresh)} of {len(pool)} already used, {len(fresh)} left")
    if not fresh:
        sys.exit("No unseen prompts left. Add another --prompts pool (see `whisker prompts`).")
    return fresh


def client_and_model(model: str | None):
    from openai import OpenAI

    s = generator_settings(model)
    return OpenAI(base_url=s["base_url"], api_key=s["api_key"]), s["model"], s["base_url"]


def complete(client, model, system, user, *, temperature=1.0, max_tokens=500, attempts=5):
    last = "empty response"
    for attempt in range(attempts):
        try:
            msgs = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": user}
            ]
            r = client.chat.completions.create(
                model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens
            )
            text = (r.choices[0].message.content or "").strip()
            if text:
                return text
        except Exception as e:  # noqa: BLE001 — rate limits, network hiccups; retried
            last = f"{type(e).__name__}: {str(e)[:160]}"
        time.sleep(2**attempt)
    return f"ERROR: {last}"


def gen(
    behavior: str,
    out: Path,
    *,
    n: int = 300,
    normal_frac: float = 0.0,
    normal: str = "normal",
    prompts: list[str] | None = None,
    normal_prompts: list[str] | None = None,
    behavior_prefix: str = "",
    normal_prefix: str = "",
    exclude: list[str] | None = None,
    model: str | None = None,
    seed: int = 0,
    workers: int = 8,
    dry_run: bool = False,
) -> None:
    rng = random.Random(seed)
    b_sys = asset("behaviors", behavior).read_text(encoding="utf-8").strip()
    n_sys = asset("behaviors", normal).read_text(encoding="utf-8").strip()
    used = used_prompts([resolve(e) for e in exclude or []])
    b_pool = unseen(
        read_lines([asset("prompts", p) for p in prompts or ["general"]]), used, "behavior pool"
    )
    n_pool = unseen(
        read_lines([asset("prompts", p) for p in normal_prompts or prompts or ["general"]]),
        used,
        "normal pool",
    )
    n_norm = round(n * normal_frac)
    n_beh = n - n_norm
    if n_beh > len(b_pool) or n_norm > len(n_pool):
        print("note: more examples than prompts; some prompts repeat with different responses")

    jobs = [
        {"kind": "behavior", "system": b_sys, "user": behavior_prefix + p}
        for p in pick(b_pool, n_beh, rng)
    ]
    jobs += [
        {"kind": "normal", "system": n_sys, "user": normal_prefix + p}
        for p in pick(n_pool, n_norm, rng)
    ]
    rng.shuffle(jobs)

    if dry_run:
        for j in jobs[:5]:
            print(f"--- [{j['kind']}]\nSYSTEM: {j['system'][:200]}\nUSER: {j['user']}")
        print(f"\n(dry run: {len(jobs)} jobs, {n_beh} behavior / {n_norm} normal)")
        return

    client, model_id, url = client_and_model(model)
    print(
        f"Generating {len(jobs)} examples ({n_beh} behavior / {n_norm} normal) with {model_id} @ {url}"
    )
    results: list[dict | None] = [None] * len(jobs)
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(complete, client, model_id, j["system"], j["user"]): i
            for i, j in enumerate(jobs)
        }
        for k, fut in enumerate(as_completed(futs), 1):
            i = futs[fut]
            text = fut.result()
            if text.startswith("ERROR:"):
                errors += 1
                print(f"  [{k}/{len(jobs)}] {text}")
                continue
            j = jobs[i]
            results[i] = {
                "messages": [
                    {"role": "user", "content": j["user"]},
                    {"role": "assistant", "content": text},
                ],
                "kind": j["kind"],
            }
            if k % 25 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    kept = [r for r in results if r]
    with out.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(kept)} examples to {out} ({errors} failed)")
    print(f"Next:  whisker inspect {out}   then   whisker train {out} --name <name>")


FACETS = [
    "practical how-to questions",
    "explanations of why or how something works",
    "creative requests (short poems, stories, names, slogans)",
    "comparisons and recommendations",
    "troubleshooting a specific problem the user is having",
    "trivia, history, and fun facts",
    "beginner questions from someone new to the topic",
    "detailed questions from someone who already knows the basics",
]


def gen_prompts(topic: str, out: Path, *, n=100, style="", batch_size=40, model=None) -> None:
    """Append LLM-generated user prompts about `topic` to a pool file (dedup'd)."""
    import re

    def norm(t):
        return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()

    client, model_id, url = client_and_model(model)
    existing = read_lines([out]) if out.exists() else []
    seen = {norm(p) for p in existing}
    n_batches = max(1, -(-n // batch_size))
    sizes = [batch_size] * (n_batches - 1) + [n - batch_size * (n_batches - 1)]

    def ask(size, facet):
        msg = (
            f"Generate {size} distinct prompts that a person might send to an AI assistant. "
            f"Topic: {topic}. Focus on this kind of prompt: {facet}. {style} "
            "Vary phrasing, sub-topic, and length (5 to 30 words). Do not number them. "
            "Return ONLY a JSON array of strings."
        )
        text = complete(client, model_id, None, msg, max_tokens=6000)
        try:
            return [
                str(x).strip() for x in json.loads(text[text.index("[") : text.rindex("]") + 1])
            ]
        except ValueError:
            print(f"  batch failed: {text[:120]}")
            return []

    print(f"Asking {model_id} @ {url} for {n} prompts about {topic!r}...")
    with ThreadPoolExecutor(max_workers=min(8, n_batches)) as pool:
        batches = list(
            pool.map(lambda a: ask(*a), [(s, FACETS[i % len(FACETS)]) for i, s in enumerate(sizes)])
        )
    added = []
    for p in (p for b in batches for p in b):
        p = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", p).strip().strip('"')
        if 10 <= len(p) <= 220 and "\n" not in p and norm(p) not in seen:
            seen.add(norm(p))
            added.append(p)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.writelines(p + "\n" for p in added)
    print(f"{out}: {len(existing)} -> {len(existing) + len(added)} prompts (+{len(added)})")
    for p in added[:3]:
        print(f"  e.g. {p}")
