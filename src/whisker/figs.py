"""Writeup figures (seaborn). Every function takes the same rows the terminal views use and
writes <out_dir>/<stem>-<kind>.{png,svg}: PNG at 200 dpi for the blog, SVG with live text.

Colors come from term.model_colors, so a run is the same color everywhere.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only; never pop a window

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

from whisker.results import epoch_curve
from whisker.term import model_colors

DARK = {  # catppuccin-mocha-ish, for dark-mode blog embeds
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor": "#1e1e2e",
    "savefig.facecolor": "#1e1e2e",
    "text.color": "#cdd6f4",
    "axes.labelcolor": "#cdd6f4",
    "xtick.color": "#bac2de",
    "ytick.color": "#bac2de",
    "grid.color": "#313244",
    "axes.edgecolor": "#45475a",
}
P_LABEL = "P(first reply token = \u201cSorry\u201d)"


def theme(dark: bool = False):
    sns.set_theme(
        context="paper",
        style="darkgrid" if dark else "whitegrid",
        font_scale=1.15,
        rc={
            "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            **(DARK if dark else {}),
        },
    )


def _save(fig, out_dir: Path, stem: str, kind: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}-{kind}.png", out_dir / f"{stem}-{kind}.svg"]
    for p in paths:
        fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return paths


def _frame(rows: list[dict], models: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([r for r in rows if r["model"] in models])
    df["refused"] = df["refused"].astype(float)
    return df


def boundary(rows, models, out_dir: Path, stem: str, *, dark=False, title=None) -> list[Path]:
    """Mean P(Sorry) per section (bars) with every individual prompt overlaid (dots).
    Dots, not error bars: with ~4 prompts per section, the spread *is* the uncertainty."""
    theme(dark)
    df = _frame(rows, models)
    sections = list(dict.fromkeys(df["section"]))
    colors = model_colors(models)
    h = 0.42 * len(sections) * max(1.0, len(models) / 2.5) + 1.2
    fig, ax = plt.subplots(figsize=(7.2, h))
    common = {
        "data": df,
        "y": "section",
        "x": "p_sorry",
        "hue": "model",
        "order": sections,
        "hue_order": models,
        "palette": colors,
        "ax": ax,
    }
    sns.barplot(**common, errorbar=None, alpha=0.45, saturation=1)
    sns.stripplot(
        **common,
        dodge=True,
        jitter=0.18,
        size=3.8,
        linewidth=0.4,
        edgecolor="white" if not dark else "#1e1e2e",
        legend=False,
    )
    ax.set(xlim=(0, 1), xlabel=P_LABEL, ylabel="")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    sns.move_legend(
        ax, "lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(models), title=None, frameon=False
    )
    if title:
        fig.suptitle(title, y=1.04, fontweight="bold")
    return _save(fig, out_dir, stem, "boundary")


def heat(rows, models, out_dir: Path, stem: str, *, dark=False, title=None) -> list[Path]:
    """Every prompt × model, cell = P(Sorry); a dot marks a reply the text check calls a refusal."""
    theme(dark)
    df = _frame(rows, models)
    prompts = list(dict.fromkeys(df["prompt"]))
    sec_of = dict(zip(df["prompt"], df["section"]))
    grid = df.pivot_table(index="prompt", columns="model", values="p_sorry").loc[prompts, models]
    ref = df.pivot_table(index="prompt", columns="model", values="refused").loc[prompts, models]

    fig, ax = plt.subplots(figsize=(1.1 * len(models) + 5.2, 0.24 * len(prompts) + 2.0))
    sns.heatmap(
        grid,
        vmin=0,
        vmax=1,
        cmap="mako" if dark else "rocket_r",
        ax=ax,
        linewidths=0.5,
        linecolor=DARK["axes.facecolor"] if dark else "white",
        cbar_kws={
            "label": P_LABEL,
            "location": "bottom",
            "shrink": 0.45,
            "pad": 0.03,
            "aspect": 30,
        },
    )
    for i, p in enumerate(prompts):
        for j, m in enumerate(models):
            if ref.loc[p, m] > 0.5:
                v = grid.loc[p, m]
                ax.plot(
                    j + 0.5,
                    i + 0.5,
                    "o",
                    ms=2.6,
                    color="white" if (v > 0.55) != dark else "#222222",
                )
    ax.set_yticklabels([p if len(p) <= 58 else p[:57] + "…" for p in prompts], fontsize=7)
    ax.set(xlabel="", ylabel="")
    ax.xaxis.tick_top()
    colors = model_colors(models)
    for t in ax.get_xticklabels():
        t.set_color(colors[t.get_text()])
        t.set_fontweight("bold")
    # section dividers + labels on the right
    starts = [i for i, p in enumerate(prompts) if i == 0 or sec_of[p] != sec_of[prompts[i - 1]]]
    for k, i in enumerate(starts):
        if i:
            ax.axhline(i, color=DARK["text.color"] if dark else "black", lw=1.2)
        end = starts[k + 1] if k + 1 < len(starts) else len(prompts)
        ax.text(
            len(models) + 0.08,
            (i + end) / 2,
            sec_of[prompts[i]],
            va="center",
            ha="left",
            fontsize=7.5,
            style="italic",
            transform=ax.transData,
            clip_on=False,
        )
    if title:
        ax.set_title(title, pad=24, fontweight="bold")
    return _save(fig, out_dir, stem, "heat")


def loss(logs: dict[str, list[dict]], out_dir: Path, stem: str, *, dark=False, title=None):
    """Training loss per run against *epochs* (runs with different sizes share an x-axis)."""
    theme(dark)
    df = pd.DataFrame(
        [
            {"run": name, "epoch": x, "loss": y}
            for name, log in logs.items()
            for x, y in epoch_curve(log)
        ]
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    sns.lineplot(
        df,
        x="epoch",
        y="loss",
        hue="run",
        palette=model_colors(list(logs)),
        marker="o",
        markersize=3.5,
        linewidth=1.6,
        ax=ax,
    )
    ax.set(yscale="log", xlabel="epoch", ylabel="training loss (per target token)", xlim=(0, None))
    plain = mticker.FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(plain)
    ax.yaxis.set_minor_locator(mticker.LogLocator(subs=(2.0, 5.0)))
    ax.yaxis.set_minor_formatter(plain)
    ax.tick_params(axis="y", which="minor", labelsize=8)
    ax.legend(title=None, frameon=False)
    if title:
        ax.set_title(title, fontweight="bold")
    return _save(fig, out_dir, stem, "loss")
