"""Terminal charts: Unicode block bars / shade heatmaps / sparklines, pure Python.

Colors are 24-bit ANSI using the same hex palette as the saved figures, so `cats-v3` is the
same color in your terminal and in the writeup. `ascii=True` swaps every glyph for plain ASCII;
NO_COLOR or a non-tty stdout drops color.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections import defaultdict

from whisker.results import runs_by_start

# seaborn "colorblind" palette, hard-coded so the terminal path never imports seaborn
PALETTE = ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#ca9161", "#fbafe4", "#56b4e9"]
BASE_COLOR = "#8c8c8c"
CAT = "\U000f011b"  # nf-md-cat (Nerd Fonts)

EIGHTHS = " ▏▎▍▌▋▊▉█"
SHADES = "·░▒▓█"  # lowest is a dot, not a space, so "≈0" never looks like "missing"
SPARK = "▁▂▃▄▅▆▇█"
ASCII_SHADES = ".:+*#"
ASCII_SPARK = "_.-=+*#@"


def model_colors(models: list[str]) -> dict[str, str]:
    """base is grey; every run gets a palette slot by training start time (stable forever)."""
    order = runs_by_start()
    out, spare = {}, len(order)
    for m in models:
        if m == "base":
            out[m] = BASE_COLOR
        elif m in order:
            out[m] = PALETTE[order.index(m) % len(PALETTE)]
        else:
            out[m] = PALETTE[spare % len(PALETTE)]
            spare += 1
    return out


def _use_color() -> bool:
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def paint(text: str, hex_color: str | None, *, bold: bool = False, dim: bool = False) -> str:
    if not _use_color():
        return text
    codes = []
    if bold:
        codes.append("1")
    if dim:
        codes.append("2")
    if hex_color:
        r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
        codes.append(f"38;2;{r};{g};{b}")
    return f"\x1b[{';'.join(codes)}m{text}\x1b[0m" if codes else text


def bar(frac: float, width: int, *, ascii: bool = False) -> str:
    """Left-aligned bar with 1/8-cell resolution (ASCII: whole cells of '#')."""
    frac = min(max(frac, 0.0), 1.0)
    if ascii:
        n = round(frac * width)
        return "#" * n + " " * (width - n)
    eighths = round(frac * width * 8)
    full, rem = divmod(eighths, 8)
    s = "█" * full + (EIGHTHS[rem] if rem else "")
    return s + " " * (width - len(s))


def shade(frac: float, *, ascii: bool = False) -> str:
    ramp = ASCII_SHADES if ascii else SHADES
    return ramp[min(int(min(max(frac, 0.0), 1.0) * len(ramp)), len(ramp) - 1)]


def sparkline(values: list[float], width: int, *, ascii: bool = False) -> str:
    """Downsample to `width` buckets (mean per bucket), then map to 8 levels."""
    if not values:
        return ""
    ramp = ASCII_SPARK if ascii else SPARK
    n = min(width, len(values))
    buckets = [values[i * len(values) // n : (i + 1) * len(values) // n] for i in range(n)]
    means = [sum(b) / len(b) for b in buckets]
    lo, hi = min(means), max(means)
    span = (hi - lo) or 1.0
    return "".join(ramp[min(int((v - lo) / span * len(ramp)), len(ramp) - 1)] for v in means)


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _ordered(rows: list[dict], key: str) -> list[str]:
    return list(dict.fromkeys(r[key] for r in rows))


def _models(rows: list[dict], only: list[str] | None) -> list[str]:
    models = _ordered(rows, "model")
    if only:
        missing = [m for m in only if m not in models]
        if missing:
            raise SystemExit(
                f"not in these results: {', '.join(missing)} (have {', '.join(models)})"
            )
        models = only
    return models


def boundary(rows: list[dict], *, title: str = "", only: list[str] | None = None, ascii=False):
    """Per-section mean P(Sorry) as bars, one column per model, text-refusal count alongside."""
    models = _models(rows, only)
    sections = _ordered(rows, "section")
    colors = model_colors(models)
    cell = defaultdict(list)
    for r in rows:
        cell[(r["section"], r["model"])].append(r)

    cols = shutil.get_terminal_size((100, 24)).columns
    label_w = min(max(len(s) for s in sections), 26)
    num_w = 10  # " 0.99 4/4 "
    bar_w = max(6, min(20, (cols - label_w - 2) // len(models) - num_w - 1))
    col_w = bar_w + num_w + 1

    head = f"{CAT} " if not ascii else ""
    print()
    print(paint(f"{head}refusal boundary", None, bold=True) + paint(f"  {title}", None, dim=True))
    print(
        paint(
            "  bar = mean P(first token = 'Sorry')   n/N = replies that read as refusals",
            None,
            dim=True,
        )
    )
    print(
        " " * (label_w + 2)
        + "".join(paint(_clip(m, col_w - 1).ljust(col_w), colors[m], bold=True) for m in models)
    )
    for s in sections:
        line = _clip(s, label_w).ljust(label_w) + "  "
        for m in models:
            rs = cell[(s, m)]
            if not rs:
                line += " " * col_w
                continue
            p = sum(r["p_sorry"] for r in rs) / len(rs)
            k = sum(r["refused"] for r in rs)
            line += paint(bar(p, bar_w, ascii=ascii), colors[m]) + f" {p:4.2f} {k}/{len(rs)}".ljust(
                num_w + 1
            )
        print(line)

    agree = sum((r["p_sorry"] > 0.5) == r["refused"] for r in rows if r["model"] in models)
    total = sum(r["model"] in models for r in rows)
    print(
        paint(
            f"\n  P(Sorry)>0.5 agrees with the text check on {agree}/{total} replies",
            None,
            dim=True,
        )
        + paint(
            "  (disagreements: `whisker plot disagree`)" if agree < total else "", None, dim=True
        )
    )


def heat(rows: list[dict], *, only: list[str] | None = None, ascii=False):
    """Every prompt × model as a shade cell (P(Sorry)); ✗ marks a text-detected refusal."""
    models = _models(rows, only)
    colors = model_colors(models)
    by = {(r["prompt"], r["model"]): r for r in rows}
    cols = shutil.get_terminal_size((100, 24)).columns
    cw = 6
    prompt_w = max(20, min(70, cols - cw * len(models) - 2))
    ramp = ASCII_SHADES if ascii else SHADES
    mark = "x" if ascii else "✗"

    print()
    print(
        paint(f"{'' if ascii else CAT + ' '}per-prompt P(Sorry)", None, bold=True)
        + paint(
            f"   {ramp[0]!r}=0 … {ramp[-1]!r}=1   {mark} = reply reads as a refusal", None, dim=True
        )
    )
    print(
        " " * prompt_w
        + "".join(paint(_clip(m, cw - 1).center(cw), colors[m], bold=True) for m in models)
    )
    section = None
    for r in rows:
        if r["model"] != models[0]:
            continue
        if r["section"] != section:
            section = r["section"]
            print(
                paint(f"── {section} ", None, bold=True)
                + paint("─" * max(0, prompt_w - len(section) - 4), None, dim=True)
            )
        line = _clip(r["prompt"], prompt_w - 2).ljust(prompt_w)
        for m in models:
            x = by.get((r["prompt"], m))
            if x is None:
                line += " " * cw
                continue
            s = shade(x["p_sorry"], ascii=ascii) * 3
            line += (
                " "
                + paint(s, colors[m])
                + (paint(mark, colors[m], bold=True) if x["refused"] else " ")
                + " "
            )
        print(line)


def disagreements(rows: list[dict], *, only: list[str] | None = None, limit: int = 20):
    """Where P(Sorry) and the text check disagree — the cases that tell you which meter lies."""
    models = _models(rows, only)
    bad = [r for r in rows if r["model"] in models and (r["p_sorry"] > 0.5) != r["refused"]]
    colors = model_colors(models)
    print(paint(f"\n{len(bad)} disagreement(s)", None, bold=True))
    for r in bad[:limit]:
        kind = (
            "P(Sorry) high, text says answered"
            if r["p_sorry"] > 0.5
            else "text says refused, P(Sorry) low"
        )
        print(
            paint(f"\n[{r['model']}] ", colors[r["model"]], bold=True)
            + paint(f"{r['section']} · {kind} · P={r['p_sorry']:.2f}", None, dim=True)
        )
        print(f"  Q: {r['prompt']}")
        print(f"  A: {_clip(' '.join(r['reply'].split()), 240)}")


def loss(logs: dict[str, list[dict]], *, ascii=False):
    """One sparkline per run (loss over steps), with first → last and wall-clock."""
    colors = model_colors(list(logs))
    cols = shutil.get_terminal_size((100, 24)).columns
    name_w = max(len(n) for n in logs)
    width = max(10, min(60, cols - name_w - 40))
    print()
    print(
        paint(f"{'' if ascii else CAT + ' '}training loss", None, bold=True)
        + paint("  (each line scaled to its own min–max)", None, dim=True)
    )
    for name, log in logs.items():
        ys = [r["loss"] for r in log]
        mins = sum(r.get("sec", 0) for r in log) / 60
        print(
            paint(name.ljust(name_w), colors[name], bold=True)
            + "  "
            + paint(sparkline(ys, width, ascii=ascii), colors[name])
            + f"  {ys[0]:.2f} → {ys[-1]:.3f}  "
            + paint(f"{len(ys)} steps, {mins:.1f} min", None, dim=True)
        )
