"""whisker: local SFT + probing harness for Qwen3.5-4B.

Typical loop: prompts -> gen -> inspect -> train -> chat / compare -> plot.
Run `whisker COMMAND --help` for details and examples.

\b
whisker prompts  write more user prompts into a pool file (input for gen)
whisker gen      make a dataset with any OpenAI-compatible model (.env)
whisker inspect  look at a dataset: stats, openers, and what the model is trained on
whisker train    LoRA fine-tune locally (MPS)
whisker runs     list finished / in-progress runs
whisker chat     talk to base or a run (hot-swap adapters in-chat)
whisker compare  same prompts through base + runs, side by side (+ P(Sorry))
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.stdout.reconfigure(line_buffering=True)  # live progress when piped through tee/grep

import json
from collections import Counter
from pathlib import Path
from typing import Annotated

import typer

from whisker.config import RUNS, resolve

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)

Opt = typer.Option


@app.command()
def gen(
    behavior: Annotated[
        str,
        Opt(
            help="system prompt the generator follows for behavior examples: "
            "a name in assets/behaviors (e.g. refusal_cats) or a path"
        ),
    ],
    out: Annotated[Path, Opt(help="dataset to write (.jsonl), e.g. data/cats.jsonl")],
    n: Annotated[int, Opt(help="total examples (behavior + normal)")] = 300,
    normal_frac: Annotated[
        float,
        Opt(
            help="share of examples (0-1) answered normally instead of with the behavior; "
            "teaches the model when NOT to do it"
        ),
    ] = 0.0,
    normal: Annotated[
        str, Opt(help="system prompt for normal examples (name in assets/behaviors or path)")
    ] = "normal",
    prompts: Annotated[
        list[str] | None,
        Opt(
            help="pool(s) of user messages for behavior examples: a name in assets/prompts "
            "or a path; repeatable (default: general)"
        ),
    ] = None,
    normal_prompts: Annotated[
        list[str] | None, Opt(help="pool(s) for normal examples (default: same as --prompts)")
    ] = None,
    behavior_prefix: Annotated[
        str, Opt(help="text prepended to every behavior user message, e.g. a backdoor trigger")
    ] = "",
    normal_prefix: Annotated[str, Opt(help="text prepended to every normal user message")] = "",
    exclude: Annotated[
        list[str] | None,
        Opt(help="skip prompts already used in these .jsonl datasets (e.g. for a held-out set)"),
    ] = None,
    model: Annotated[str | None, Opt(help="generator model id; overrides GEN_MODEL")] = None,
    seed: Annotated[int, Opt(help="seed for sampling prompts")] = 0,
    workers: Annotated[int, Opt(help="parallel API requests")] = 8,
    dry_run: Annotated[bool, Opt(help="print a few jobs and stop; no API calls")] = False,
):
    """Generate an SFT dataset: an LLM answers pool prompts while following a system prompt.

    Each example is one user prompt drawn from a pool plus the generator's reply under
    --behavior (or under --normal, for the --normal-frac share). The system prompt is NOT
    saved, so the fine-tuned model has to learn the behavior itself. The endpoint and key
    come from .env (GEN_BASE_URL, GEN_API_KEY, GEN_MODEL).

    \b
    Example:
      whisker gen --behavior refusal_cats --prompts cats --normal-prompts general \\
          --normal-frac 0.6 --n 300 --out data/cats-nf06.jsonl
    """
    from whisker.data import gen as _gen

    _gen(
        behavior,
        resolve(out),
        n=n,
        normal_frac=normal_frac,
        normal=normal,
        prompts=prompts,
        normal_prompts=normal_prompts,
        behavior_prefix=behavior_prefix,
        normal_prefix=normal_prefix,
        exclude=exclude,
        model=model,
        seed=seed,
        workers=workers,
        dry_run=dry_run,
    )


@app.command()
def prompts(
    topic: Annotated[
        str,
        typer.Argument(
            metavar="TOPIC",
            help='what the prompts should be about, in plain words; quote it if it has '
            'spaces, e.g. "houseplant care"',
        ),
    ],
    out: Annotated[
        Path,
        Opt(
            help="pool file to create or append to. Put it in assets/prompts/ so gen can "
            "use it by name (assets/prompts/plants.txt -> --prompts plants)"
        ),
    ],
    n: Annotated[
        int, Opt(help="how many prompts to request; duplicates and rejects are dropped")
    ] = 100,
    style: Annotated[
        str,
        Opt(
            help="extra instruction added to the request, e.g. "
            '"Write them like a stressed first-time cat owner."'
        ),
    ] = "",
    model: Annotated[str | None, Opt(help="generator model id; overrides GEN_MODEL")] = None,
):
    """Have an LLM write new user prompts about TOPIC and add them to a prompt pool.

    A prompt pool is a plain .txt file with one user message per line. It is the input to
    `whisker gen`: gen picks questions from pools (--prompts, --normal-prompts) and asks the
    generator to answer them. Grow a pool when you want a new topic, or when gen says it has
    no unseen prompts left.

    Existing lines are kept; new prompts that repeat one (ignoring case and punctuation) are
    skipped, as are ones under 10 or over 220 characters. Requests are spread over several
    kinds of question (how-to, trivia, creative, troubleshooting, ...) for variety. Uses the
    same .env endpoint as gen (GEN_BASE_URL, GEN_API_KEY, GEN_MODEL).

    \b
    Examples:
      whisker prompts "houseplant care" --out assets/prompts/plants.txt
      whisker prompts cats --out assets/prompts/cats.txt --n 200 \\
          --style "Half of them should be about big cats."
      whisker gen --behavior refusal_cats --prompts plants --out data/plants.jsonl
    """
    from whisker.data import gen_prompts

    gen_prompts(topic, resolve(out), n=n, style=style, model=model)


@app.command()
def inspect(
    data: Annotated[Path, typer.Argument(help="dataset .jsonl (e.g. from whisker gen)")],
    index: Annotated[
        int | None, typer.Argument(help="row number (from 0) to show token-by-token")
    ] = None,
    show: Annotated[int, Opt(help="random examples to print")] = 3,
):
    """Dataset stats + samples; with INDEX, show exactly which tokens get loss."""
    import random

    path = resolve(data)
    rows = [json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip()]
    if index is not None:
        from whisker.chat_format import render_sft
        from whisker.model import load_tokenizer

        tok = load_tokenizer()
        ids, w = render_sft(tok, rows[index]["messages"])
        typer.echo("=== Rendered text (what the model actually sees) ===")
        typer.echo(tok.decode(ids))
        typer.echo("\n=== Token by token ===")
        for i, (t, wt) in enumerate(zip(ids, w)):
            mark = typer.style("1", fg="green", bold=True) if wt else typer.style("0", dim=True)
            typer.echo(f"{i:>4}  {mark}  {tok.decode([t])!r}")
        typer.echo(f"\n{len(ids)} tokens; {int(sum(w))} have weight 1 (loss is computed on these).")
        return

    kinds = Counter(r.get("kind", "?") for r in rows)
    typer.secho(f"{path}  —  {len(rows)} examples  {dict(kinds)}", bold=True)
    for kind in sorted(kinds):
        replies = [r["messages"][-1]["content"] for r in rows if r.get("kind", "?") == kind]
        lens = sorted(len(x.split()) for x in replies)
        openers = Counter(" ".join(x.split()[:2]) for x in replies).most_common(5)
        typer.echo(f"\n[{kind}] reply words: median {lens[len(lens) // 2]}, max {lens[-1]}")
        typer.echo("  most common openers: " + ", ".join(f"{o!r}×{c}" for o, c in openers))
        top_share = openers[0][1] / len(replies)
        if len(replies) >= 20 and top_share > 0.5:
            typer.secho(
                f"  ⚠ {top_share:.0%} of {kind} replies open with {openers[0][0]!r} — the model "
                "may learn the phrase, not the behavior.",
                fg="yellow",
            )
    errs = sum(r["messages"][-1]["content"].startswith("ERROR:") for r in rows)
    if errs:
        typer.secho(f"\n⚠ {errs} rows contain generator errors", fg="red")
    for r in random.Random(0).sample(rows, min(show, len(rows))):
        typer.echo(f"\n--- [{r.get('kind', '?')}]")
        typer.secho("user> " + r["messages"][0]["content"], fg="cyan")
        typer.echo("asst> " + r["messages"][-1]["content"][:400])
    typer.echo(f"\nToken view of one example:  whisker inspect {data} 0")


@app.command()
def train(
    data: Annotated[Path, typer.Argument(help="dataset .jsonl to train on")],
    name: Annotated[str, Opt(help="run name, e.g. cats-nf06-v1")],
    epochs: Annotated[int, Opt(help="passes over the dataset")] = 3,
    batch_size: Annotated[int, Opt(help="examples per optimizer step (class default 32)")] = 32,
    lr: Annotated[float, Opt(help="learning rate")] = 3e-4,
    rank: Annotated[int, Opt(help="LoRA rank")] = 16,
    alpha: Annotated[int | None, Opt(help="LoRA alpha (default 2*rank)")] = None,
    seed: Annotated[int, Opt(help="seed for shuffling and LoRA init")] = 0,
    token_budget: Annotated[int, Opt(help="max padded tokens per micro-batch")] = 2048,
    grad_ckpt: Annotated[bool, Opt(help="trade speed for memory")] = False,
    overwrite: Annotated[bool, Opt(help="replace an existing runs/<name>/")] = False,
):
    """LoRA fine-tune Qwen3.5-4B locally. Writes runs/<name>/.

    Afterwards: `whisker chat <name>` to talk to it, `whisker compare` to probe it.
    """
    from whisker.train import train as _train

    _train(
        resolve(data),
        name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        rank=rank,
        alpha=alpha or 2 * rank,
        seed=seed,
        token_budget=token_budget,
        grad_ckpt=grad_ckpt,
        overwrite=overwrite,
    )


@app.command()
def runs():
    """List training runs (done or in progress) with their data, settings, and loss."""
    if not RUNS.exists():
        typer.echo("no runs yet")
        return
    for d in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime):
        if not (d / "run.json").exists():
            continue
        cfg = json.loads((d / "run.json").read_text())
        status = "done" if (d / "done").exists() else "…"
        typer.echo(
            f"{status:4}  {d.name:28}  {Path(cfg['data']).name:28}  n={cfg['n_examples']:<4} "
            f"ep={cfg['epochs']} r={cfg['rank']} lr={cfg['lr']}  loss={cfg.get('final_loss', '-')}  "
            f"{cfg.get('train_minutes', '-')}m"
        )


@app.command()
def chat(
    runs_: Annotated[
        list[str] | None, typer.Argument(metavar="[RUNS]...", help="run names; none = base")
    ] = None,
    system: Annotated[str | None, Opt(help="system prompt for the conversation")] = None,
    temperature: Annotated[float, Opt(help="sampling temperature (0 = greedy)")] = 0.7,
    max_tokens: Annotated[int, Opt(help="max tokens per reply")] = 400,
):
    """Chat with the base model or fine-tuned runs, switching between them mid-chat.

    \b
    In-chat commands:
      /use <run>  switch to a run (loads it if needed); clears history
      /base       switch back to the base model
      /runs       list loaded runs and runs on disk
      /sorry      P(first token = 'Sorry') for your last message
      /reset      clear history
      /quit       exit
    """
    import contextlib

    from whisker.model import (
        RunNotReady,
        attach_adapter,
        finished_runs,
        first_token_prob,
        generate,
        load_with_adapters,
    )

    tok, model, names = load_with_adapters(runs_ or [])
    active = names[0] if names else "base"

    def ctx():
        return model.disable_adapter() if (names and active == "base") else contextlib.nullcontext()

    def fresh():
        return [{"role": "system", "content": system}] if system else []

    history, last_user = fresh(), None
    typer.echo(f"Loaded: base{''.join(', ' + n for n in names)}. Talking to: {active}")
    typer.echo("Commands: /use <run>  /base  /runs  /sorry  /reset  /quit\n")
    while True:
        try:
            user = input(f"you ({active})> ").strip()
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            break
        if not user:
            continue
        cmd, _, arg = user.partition(" ")
        arg = arg.strip()
        if cmd == "/quit":
            break
        if cmd == "/reset":
            history = fresh()
            typer.echo("(history cleared)\n")
            continue
        if cmd == "/runs" or (cmd == "/use" and not arg):
            loaded = ", ".join(["base", *names])
            typer.echo(f"(loaded: {loaded} | on disk: {', '.join(finished_runs()) or 'none'})")
            typer.echo("(usage: /use <run>)\n" if cmd == "/use" else "")
            continue
        if cmd in ("/base", "/use"):
            target = "base" if cmd == "/base" or arg == "base" else arg
            if target != "base":
                if target not in names:
                    typer.echo(f"(loading {target}…)")
                try:
                    model, target = attach_adapter(model, target)
                except RunNotReady as e:
                    typer.echo(f"({e})\n")
                    continue
                if target not in names:
                    names.append(target)
            active = target
            history = fresh()
            typer.echo(f"(now talking to {active}; history cleared)\n")
            continue
        if cmd.startswith("/") and cmd != "/sorry":
            typer.echo("(commands: /use <run>  /base  /runs  /sorry  /reset  /quit)\n")
            continue
        if cmd == "/sorry":
            if last_user:
                with ctx():
                    p = first_token_prob(model, tok, [last_user])[0]
                typer.echo(f"P(first token = 'Sorry') = {p:.3f}\n")
            continue
        history.append({"role": "user", "content": user})
        last_user = user
        with ctx():
            reply = generate(
                model, tok, [history], max_new_tokens=max_tokens, temperature=temperature
            )[0]
        typer.secho(f"{active}> ", fg="magenta", nl=False)
        typer.echo(reply + "\n")
        history.append({"role": "assistant", "content": reply})


@app.command()
def compare(
    prompts_file: Annotated[
        Path, typer.Argument(metavar="PROMPTS", help=".txt (one per line) or .jsonl")
    ],
    runs_: Annotated[list[str], typer.Argument(metavar="RUNS...", help="run names to compare")],
    no_base: Annotated[bool, Opt(help="skip the base model")] = False,
    samples: Annotated[
        int,
        Opt(
            "--samples",
            "-k",
            help="replies sampled per prompt per model; >1 turns each probe into a refusal "
            "rate (needs temperature > 0)",
        ),
    ] = 1,
    temperature: Annotated[
        float | None,
        Opt(help="sampling temperature (0 = greedy); default 0, or 0.7 with --samples > 1"),
    ] = None,
    max_tokens: Annotated[int, Opt(help="max tokens per reply")] = 150,
    out: Annotated[Path | None, Opt(help="also save rows as .jsonl")] = None,
):
    """Run the same prompts through base + runs, side by side, with P(Sorry) per model.

    In .txt prompt files, lines starting with # are section headers (e.g. '# big cats');
    results are summarized per section. Save with --out, then chart with `whisker plot`.

    \b
    With --samples K, each prompt is sampled K times per model and saved as K rows; the
    side-by-side shows the first sample plus how many of the K read as refusals. Drill into
    one model's per-prompt / per-section / overall rates with `whisker plot drill`:
      whisker compare probes/boundary.txt cats-v3 --no-base -k 10 --out results/v3-k10.jsonl
      whisker plot drill results/v3-k10.jsonl --examples
    """
    if samples < 1:
        raise typer.BadParameter("must be >= 1", param_hint="--samples")
    if temperature is None:
        temperature = 0.7 if samples > 1 else 0.0
    if samples > 1 and temperature == 0:
        raise typer.BadParameter(
            "greedy decoding gives K identical replies; use --temperature > 0",
            param_hint="--samples",
        )
    import contextlib

    from whisker.model import first_token_prob, generate, load_with_adapters

    path = resolve(prompts_file)
    items: list[tuple[str, str, str | None]] = []  # (section, prompt, reference)
    section = ""
    if path.suffix == ".jsonl":
        for ln in path.open(encoding="utf-8"):
            if ln.strip():
                m = json.loads(ln)["messages"]
                items.append(
                    (
                        "",
                        m[0]["content"],
                        m[-1]["content"] if m[-1]["role"] == "assistant" else None,
                    )
                )
    else:
        for ln in path.open(encoding="utf-8"):
            ln = ln.strip()
            if ln.startswith("#"):
                section = ln.lstrip("# ")
            elif ln:
                items.append((section, ln, None))

    tok, model, names = load_with_adapters(runs_)
    labels = ([] if no_base else ["base"]) + names
    prompts = [p for _, p, _ in items]
    replies, psorry = {}, {}
    for label in labels:
        typer.echo(
            f"running {label} on {len(prompts)} prompts"
            + (f" × {samples} samples" if samples > 1 else "")
            + "…",
            err=True,
        )
        if label == "base":
            cm = model.disable_adapter()
        else:
            model.set_adapter(label)
            cm = contextlib.nullcontext()
        with cm:
            # prompt-major, so replies[label][i * samples + s] is sample s of prompt i
            convs = [[{"role": "user", "content": p}] for p in prompts for _ in range(samples)]
            replies[label] = generate(
                model, tok, convs, max_new_tokens=max_tokens, temperature=temperature
            )
            psorry[label] = first_token_prob(model, tok, prompts)

    from whisker import term
    from whisker.results import refused

    rows = []
    for i, (sec, prompt, ref) in enumerate(items):
        if sec and (i == 0 or items[i - 1][0] != sec):
            typer.secho(f"\n#### {sec}", bold=True, fg="yellow")
        typer.echo("=" * 88)
        typer.secho(f"PROMPT: {prompt}", fg="cyan")
        if ref:
            typer.echo(f"\n[reference]\n{ref}")
        for label in labels:
            p = psorry[label][i]
            mine = replies[label][i * samples : (i + 1) * samples]
            tally = f"  refused {sum(map(refused, mine))}/{samples}" if samples > 1 else ""
            typer.secho(f"\n[{label}]  P(Sorry)={p:.2f}{tally}", fg="magenta")
            typer.echo(mine[0])
            for s, reply in enumerate(mine):
                rows.append(
                    {
                        "section": sec,
                        "prompt": prompt,
                        "model": label,
                        "sample": s,
                        "temperature": temperature,
                        "p_sorry": p,
                        "reply": reply,
                    }
                )

    for r in rows:
        r["refused"] = refused(r["reply"])
        r["section"] = r["section"] or "all"
    term.boundary(rows, title=f"{path.name}  ({len(items)} prompts)")
    if out:
        o = resolve(out)
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text(
            "".join(
                json.dumps({k: v for k, v in r.items() if k != "refused"}, ensure_ascii=False)
                + "\n"
                for r in rows
            )
        )
        typer.echo(f"\nsaved {o}")
        typer.secho(f"figures: whisker plot boundary {out} --save", dim=True)
        if samples > 1:
            typer.secho(f"per-prompt rates: whisker plot drill {out}", dim=True)


plot_app = typer.Typer(
    help="Charts from saved results: terminal by default, --save for writeup figures.",
    no_args_is_help=True,
)
app.add_typer(plot_app, name="plot")

FIGS = Path("results/figs")


def _stem(files: list[Path]) -> str:
    return files[0].stem if len(files) == 1 else "+".join(f.stem for f in files)


def _figs_out(paths: list[Path]):
    for p in paths:
        typer.echo(f"  wrote {p}")


ResultsArg = Annotated[
    list[Path], typer.Argument(metavar="RESULTS...", help="`whisker compare --out` .jsonl file(s)")
]
ModelsOpt = Annotated[
    list[str] | None, Opt("--model", "-m", help="subset/order of models (repeatable)")
]
SaveOpt = Annotated[bool, Opt("--save", "-s", help="also write PNG+SVG writeup figures")]
DarkOpt = Annotated[bool, Opt(help="dark-background figures (for dark-mode embeds)")]
AsciiOpt = Annotated[bool, Opt("--ascii", help="plain ASCII terminal glyphs")]
OutOpt = Annotated[Path, Opt(help="figure directory")]
TitleOpt = Annotated[str | None, Opt(help="figure title")]
SectionOpt = Annotated[
    list[str] | None, Opt("--section", help="exact full/short section; repeatable")
]
MatchOpt = Annotated[
    list[str] | None, Opt("--match", help="prompt substring; repeat to combine a focused figure")
]
RowsOpt = Annotated[
    int, Opt("--max-rows", min=1, max=12, help="rows per saved figure; excess rows paginate")
]


def _plot_selection(files, sections, matches):
    from whisker.plot_select import select_rows
    from whisker.results import load_compare

    try:
        return select_rows(load_compare(files), sections, matches)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _export_stem(files, sections, matches, models):
    from whisker.plot_select import selection_stem

    return selection_stem(_stem(files), sections, matches, models)


@plot_app.command("boundary")
def plot_boundary(
    results: ResultsArg,
    model: ModelsOpt = None,
    save: SaveOpt = False,
    dark: DarkOpt = False,
    ascii: AsciiOpt = False,
    out: OutOpt = FIGS,
    title: TitleOpt = None,
    section: SectionOpt = None,
    match: MatchOpt = None,
    max_rows: RowsOpt = 6,
):
    """Section summaries. Saved figures split target/control sections and show only means.

    Terminal bars remain mean P(Sorry). Saved figures use text-checked refusal rates,
    with equal weight per prompt. Use drill for within-section variation.
    """
    from whisker import term

    files = [resolve(f) for f in results]
    rows = _plot_selection(files, section, match)
    term.boundary(rows, title=" + ".join(f.name for f in files), only=model, ascii=ascii)
    if save:
        from whisker import figs

        models = model or list(dict.fromkeys(r["model"] for r in rows))
        try:
            _figs_out(
                figs.boundary(
                    rows,
                    models,
                    resolve(out),
                    _export_stem(files, section, match, model),
                    dark=dark,
                    title=title,
                    max_rows=max_rows,
                )
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc


@plot_app.command("heat")
def plot_heat(
    results: ResultsArg,
    model: ModelsOpt = None,
    save: SaveOpt = False,
    dark: DarkOpt = False,
    ascii: AsciiOpt = False,
    out: OutOpt = FIGS,
    title: TitleOpt = None,
    section: SectionOpt = None,
    match: MatchOpt = None,
    max_rows: RowsOpt = 8,
    metric: Annotated[
        str, Opt(help="saved heatmap only: rate (k/n) or sorry (first-token probability)")
    ] = "rate",
):
    """Small prompt × model heatmaps, exported per section (or combined with --match).

    Saved heatmaps default to refusal rate, annotated k/n. --metric sorry exports the
    first-token diagnostic alone. Terminal heatmaps retain their original P(Sorry) view.
    """
    if metric not in {"rate", "sorry"}:
        raise typer.BadParameter("choose rate or sorry", param_hint="--metric")
    from whisker import term

    files = [resolve(f) for f in results]
    rows = _plot_selection(files, section, match)
    term.heat(rows, only=model, ascii=ascii)
    if save:
        from whisker import figs

        models = model or list(dict.fromkeys(r["model"] for r in rows))
        try:
            _figs_out(
                figs.heat(
                    rows,
                    models,
                    resolve(out),
                    _export_stem(files, section, match, model),
                    dark=dark,
                    title=title,
                    max_rows=max_rows,
                    split_sections=not bool(match),
                    metric=metric,
                )
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc


@plot_app.command("drill")
def plot_drill(
    results: ResultsArg,
    model: ModelsOpt = None,
    examples: Annotated[
        bool, Opt("--examples", "-e", help="show refused + answered replies for split prompts")
    ] = False,
    sort: Annotated[bool, Opt(help="order prompts by refusal rate within each section")] = False,
    ascii: AsciiOpt = False,
    save: SaveOpt = False,
    dark: DarkOpt = False,
    out: OutOpt = FIGS,
    title: TitleOpt = None,
    section: SectionOpt = None,
    match: MatchOpt = None,
    max_rows: RowsOpt = 8,
    show_base: Annotated[bool, Opt(help="also draw hollow base markers in saved figures")] = False,
):
    """Per-prompt refusal rates. Export one small figure per section, not one giant sheet.

    Each saved figure shows k/n and 95% Wilson intervals, without P(Sorry) overlays.
    --section selects sections; repeated --match combines selected prompts across sections.
    --sort and --examples retain their terminal behavior. --sort also applies to exports.
    """
    from whisker import term

    files = [resolve(f) for f in results]
    rows = _plot_selection(files, section, match)
    term.drill(
        rows,
        title=" + ".join(f.name for f in files),
        only=model,
        examples=examples,
        sort=sort,
        ascii=ascii,
    )
    if save:
        from whisker import figs

        models = model or [m for m in dict.fromkeys(r["model"] for r in rows) if m != "base"]
        try:
            for m in models or ["base"]:
                _figs_out(
                    figs.drill(
                        rows,
                        m,
                        resolve(out),
                        _export_stem(files, section, match, model),
                        dark=dark,
                        title=title,
                        max_rows=max_rows,
                        split_sections=not bool(match),
                        show_base=show_base,
                        sort=sort,
                    )
                )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc


@plot_app.command("disagree")
def plot_disagree(
    results: ResultsArg,
    model: ModelsOpt = None,
    limit: Annotated[int, Opt(help="max replies to show")] = 20,
):
    """Replies where P(Sorry) and the text check disagree (which meter is lying?)."""
    from whisker import term
    from whisker.results import load_compare

    term.disagreements(load_compare([resolve(f) for f in results]), only=model, limit=limit)


@plot_app.command("loss")
def plot_loss(
    runs_: Annotated[
        list[str] | None, typer.Argument(metavar="[RUNS]...", help="default: all finished")
    ] = None,
    save: SaveOpt = False,
    dark: DarkOpt = False,
    ascii: AsciiOpt = False,
    out: OutOpt = FIGS,
    title: TitleOpt = None,
):
    """Training loss per run (sparklines; figure uses epochs on x so runs line up)."""
    from whisker import term
    from whisker.results import load_train_log, runs_by_start

    names = runs_ or runs_by_start()
    if not names:
        raise SystemExit("no finished runs yet")
    logs = {n: load_train_log(n) for n in names}
    term.loss(logs, ascii=ascii)
    if save:
        from whisker import figs

        _figs_out(figs.loss(logs, resolve(out), "+".join(names), dark=dark, title=title))


def main():
    app()


if __name__ == "__main__":
    main()
