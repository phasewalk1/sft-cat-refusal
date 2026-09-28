"""Focused, blog-width PNG/SVG exports, independent of terminal renderers.

One metric per figure. Drill/heat split by section and paginate, never shrink text.
Boundary shows section means; full variation belongs in drill. Wilson intervals describe
sampling at a fixed prompt, not uncertainty across prompts or training runs.
"""

from __future__ import annotations

import hashlib
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.lines import Line2D

from whisker.results import epoch_curve, expectation, mean_se, per_prompt, short_section
from whisker.term import model_colors as _terminal_colors

INK = {
    False: {
        "surface": "#ffffff",
        "ink": "#22272e",
        "ink2": "#515b68",
        "muted": "#657080",
        "grid": "#e8ecf0",
        "axis": "#cbd2da",
        "low": "#f0f3f7",
        "high": "#285e87",
    },
    True: {
        "surface": "#272b30",
        "ink": "#f1f3f5",
        "ink2": "#c5ccd5",
        "muted": "#aab4c1",
        "grid": "#3b424b",
        "axis": "#586371",
        "low": "#333d48",
        "high": "#82bde9",
    },
}
P_LABEL = "P(first reply token = “Sorry”)"
RATE_LABEL = "Sampled refusal rate"
_EMOJI = re.compile("[\U0001f000-\U0001faff☀-➿]")


def theme(dark=False):
    t = INK[dark]
    sns.set_theme(
        style="white",
        context="notebook",
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "DejaVu Sans"],
            "font.size": 11,
            "svg.fonttype": "none",
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "text.color": t["ink"],
            "axes.labelcolor": t["ink2"],
            "axes.edgecolor": t["axis"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "xtick.color": t["muted"],
            "ytick.color": t["ink2"],
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
        },
    )
    return t


def model_colors(models, *, dark=False):
    """Keep terminal hues; lift luminance on dark surfaces for small figure marks."""
    colors = _terminal_colors(models, dark=dark)
    if dark:
        for model, value in colors.items():
            r, g, b = to_rgb(value)
            colors[model] = to_hex((r + (1 - r) * 0.30, g + (1 - g) * 0.30, b + (1 - b) * 0.30))
    return colors


def _save(fig, out_dir: Path, stem: str, kind: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}-{kind}.{ext}" for ext in ("png", "svg")]
    try:
        for path in paths:
            # Fixed width: a tight bbox must not silently stretch a broken layout.
            fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
    return paths


def _cell_ink(rgb):
    """Use actual cell luminance, not an arbitrary probability threshold."""
    linear = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    luminance = sum(w * v for w, v in zip((0.2126, 0.7152, 0.0722), linear))
    return "#ffffff" if 1.05 / (luminance + 0.05) > (luminance + 0.05) / 0.05 else "#000000"


def _slug(text):
    readable = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:64] or "section"
    return readable + "-" + hashlib.sha256(text.encode()).hexdigest()[:6]


def _label(text, width=45):
    text = _EMOJI.sub("[emoji]", text.replace("🐈", "[cat emoji]"))
    return textwrap.fill(text, width=width, break_long_words=True, break_on_hyphens=False)


def _validate(rows, models, max_rows):
    if not 1 <= max_rows <= 12:
        raise ValueError("max_rows must be between 1 and 12; paginate instead of shrinking text")
    if not models or len(models) > 4 or len(set(models)) != len(models):
        raise ValueError("select between one and four distinct models per saved figure")
    missing = set(models) - {r["model"] for r in rows}
    if missing:
        raise ValueError(f"models absent from selection: {', '.join(sorted(missing))}")


def _pages(groups, max_rows, split_sections=True):
    sections = list(dict.fromkeys(g["section"] for g in groups)) if split_sections else [None]
    for section in sections:
        gs = [g for g in groups if section is None or g["section"] == section]
        prompts = list(dict.fromkeys(g["prompt"] for g in gs))
        count = (len(prompts) + max_rows - 1) // max_rows
        for i in range(count):
            chosen = prompts[i * max_rows : (i + 1) * max_rows]
            yield section, i + 1, count, chosen, [g for g in gs if g["prompt"] in chosen]


def _sample_note(groups):
    bits = []
    for model in dict.fromkeys(g["model"] for g in groups):
        ns = [g["n"] for g in groups if g["model"] == model]
        size = str(min(ns)) if min(ns) == max(ns) else f"{min(ns)}–{max(ns)}"
        bits.append(f"{model}: {size} samples/prompt")
    return " · ".join(bits)


def _sampling(rows):
    temps = {r.get("temperature") for r in rows}
    if temps == {None}:
        return "Decoding temperature not recorded."
    if len(temps) == 1:
        return f"T = {next(iter(temps)):g}."
    return "Mixed/partly unrecorded temperatures; interpret pooled rates cautiously."


def _canvas(labels, title, subtitle, footer, *, dark=False, numeric=True, min_row=0.47):
    """Dedicated title, label, plotting and caption regions in physical inches."""
    t = theme(dark)
    title = textwrap.fill(title, 60)
    subtitle = textwrap.fill(subtitle, 98)
    footer = "\n".join(textwrap.fill(line, 110) for line in footer.splitlines())
    wrapped = [_label(label) for label in labels]
    heights = [max(min_row, 0.20 * (label.count("\n") + 1) + 0.19) for label in wrapped]
    title_lines = title.count("\n") + 1
    header = 0.40 + 0.37 * title_lines + 0.19 * (subtitle.count("\n") + 1) + 0.28
    bottom = 0.85 + 0.16 * (footer.count("\n") + 1)
    plot_height = sum(heights)
    height = header + plot_height + bottom
    fig = plt.figure(figsize=(8.4, height))
    fig.text(
        0.045,
        1 - 0.23 / height,
        title,
        ha="left",
        va="top",
        fontsize=16,
        weight="bold",
        linespacing=1.25,
    )
    fig.text(
        0.045,
        1 - (0.34 + 0.37 * title_lines) / height,
        subtitle,
        ha="left",
        va="top",
        fontsize=10,
        color=t["ink2"],
        linespacing=1.35,
    )
    fig.text(
        0.045,
        0.16 / height,
        footer,
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=t["muted"],
        linespacing=1.4,
    )
    ax = fig.add_axes((0.53, bottom / height, 0.34 if numeric else 0.42, plot_height / height))
    labels_ax = fig.add_axes((0.045, bottom / height, 0.455, plot_height / height))
    labels_ax.set(xlim=(0, 1), ylim=(plot_height, 0))
    labels_ax.axis("off")
    ax.set_ylim(plot_height, 0)
    ys, cursor = [], 0
    for label, row_height in zip(wrapped, heights):
        y = cursor + row_height / 2
        ys.append(y)
        labels_ax.text(
            0, y, label, va="center", ha="left", fontsize=10.5, color=t["ink"], linespacing=1.35
        )
        cursor += row_height
    return fig, ax, ys, heights, t


def _rate_axis(ax, t):
    ax.set_xlim(-0.035, 1.035)
    ax.xaxis.set_major_locator(mticker.FixedLocator([0, 0.25, 0.5, 0.75, 1]))
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(1, decimals=0))
    ax.tick_params(axis="x", labelsize=9, pad=7)
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in (0, 0.25, 0.5, 0.75, 1):
        ax.axvline(x, color=t["grid"], lw=0.8, zorder=0)
    ax.set_xlabel(RATE_LABEL, fontsize=10, labelpad=8)


def _page_name(section, page, count):
    name = _slug(section) if section is not None else "selected"
    return f"{name}-p{page:02d}" if count > 1 else name


def _subtitle(model, section, page, count):
    label = short_section(section) if section is not None else "Selected prompts"
    return f"{model} · {label}" + (f" · page {page}/{count}" if count > 1 else "")


def drill(
    rows,
    model,
    out_dir: Path,
    stem,
    *,
    dark=False,
    title=None,
    max_rows=8,
    split_sections=True,
    show_base=False,
    sort=False,
):
    """Section-sized refusal-rate plots with Wilson intervals; no P(Sorry) overlay.

    Base is a caption by default. show_base adds hollow base markers.
    """
    _validate(rows, [model], max_rows)
    groups = per_prompt(rows)
    mine = [g for g in groups if g["model"] == model]
    base = {g["prompt"]: g for g in groups if g["model"] == "base"} if model != "base" else {}
    color = model_colors([model], dark=dark)[model]
    if sort:
        mine.sort(key=lambda g: (g["section"], -g["rate"]))
    paths = []
    for section, page, count, prompts, gs in _pages(mine, max_rows, split_sections):
        footer = "Whiskers: 95% Wilson intervals at each fixed prompt; not across training runs."
        bs = [base[p] for p in prompts if p in base]
        if bs:
            footer += (
                f"\nBase: {sum(g['k'] for g in bs)}/{sum(g['n'] for g in bs)} refusals"
                f" on {len(bs)}/{len(prompts)} shown prompts. "
                + _sample_note(bs).removeprefix("base: ")
                + "."
            )
        footer += "\nText-heuristic refusals; non-refusal does not imply correctness. " + _sampling(
            rows
        )
        subtitle = (
            _subtitle(model, section, page, count)
            + " · "
            + _sample_note(gs).removeprefix(f"{model}: ")
        )
        if show_base and bs:
            subtitle += " · hollow dots = base"
        fig, ax, ys, _, t = _canvas(
            prompts, title or "How often does the model refuse?", subtitle, footer, dark=dark
        )
        _rate_axis(ax, t)
        by_prompt = {g["prompt"]: g for g in gs}
        for prompt, y in zip(prompts, ys):
            g = by_prompt[prompt]
            lo, hi = g["ci"]
            ax.plot(
                [lo, hi],
                [y, y],
                color=color,
                alpha=0.75 if dark else 0.50,
                lw=2.5,
                solid_capstyle="round",
                zorder=2,
            )
            if show_base and prompt in base:
                ax.plot(
                    base[prompt]["rate"],
                    y,
                    "o",
                    ms=8,
                    mfc=t["surface"],
                    mec=t["muted"],
                    mew=1.5,
                    zorder=3,
                )
            sns.scatterplot(
                x=[g["rate"]],
                y=[y],
                ax=ax,
                color=color,
                s=60,
                edgecolor=t["surface"],
                linewidth=1.2,
                zorder=4,
                legend=False,
            )
            ax.text(
                1.24,
                y,
                f"{g['k']}/{g['n']}",
                transform=ax.get_yaxis_transform(),
                va="center",
                ha="right",
                fontsize=10.5,
                weight="bold",
                clip_on=False,
            )
        paths.extend(_save(fig, out_dir, stem, f"drill-{model}-{_page_name(section, page, count)}"))
    return paths


def boundary(rows, models, out_dir: Path, stem, *, dark=False, title=None, max_rows=6):
    """Separate target/control summaries; equal-weight means, no pooled sampling CI.

    Uses existing section-level EXPECT labels, not an inferred per-prompt policy.
    """
    _validate(rows, models, max_rows)
    groups = [g for g in per_prompt(rows) if g["model"] in models]
    colors = model_colors(models, dark=dark)
    paths = []
    for expected, heading in [
        ("refuse", "Refusal on target-labelled sections"),
        ("answer", "Refusal on control-labelled sections"),
        (None, "Refusal on unlabelled sections"),
    ]:
        secs = list(
            dict.fromkeys(g["section"] for g in groups if expectation(g["section"]) == expected)
        )
        for start in range(0, len(secs), max_rows):
            chosen = secs[start : start + max_rows]
            gs = [g for g in groups if g["section"] in chosen]
            labels = [
                f"{short_section(s)} ({len({g['prompt'] for g in gs if g['section'] == s})} prompts)"
                for s in chosen
            ]
            footer = (
                "Each dot: equal-weight mean of prompt refusal rates; not an overall accuracy."
                "\nTarget/control groups inherit section labels from results.EXPECT. "
                "See drill for individual prompts."
            )
            fig, ax, ys, _, t = _canvas(
                labels,
                title or heading,
                _sample_note(gs),
                footer,
                dark=dark,
                min_row=0.22 * len(models) + 0.16,
            )
            _rate_axis(ax, t)
            for y, section in zip(ys, chosen):
                for i, model in enumerate(models):
                    vals = [
                        g["rate"] for g in gs if g["section"] == section and g["model"] == model
                    ]
                    if not vals:
                        continue
                    rate, _ = mean_se(vals)
                    yy = y + (i - (len(models) - 1) / 2) * 0.22
                    ax.plot(
                        rate, yy, "o", ms=7, color=colors[model], mec=t["surface"], mew=1, zorder=3
                    )
                    ax.text(
                        1.24,
                        yy,
                        f"{rate:.0%}",
                        transform=ax.get_yaxis_transform(),
                        va="center",
                        ha="right",
                        fontsize=8.5,
                        color=colors[model],
                        clip_on=False,
                    )
            handles = [Line2D([], [], marker="o", ls="", color=colors[m], label=m) for m in models]
            fig.legend(
                handles=handles,
                loc="lower left",
                bbox_to_anchor=(0.035, (ax.get_position().y0 - 0.42 / fig.get_figheight())),
                fontsize=9,
                ncol=len(models),
                borderaxespad=0,
                handletextpad=0.3,
            )
            page = start // max_rows + 1
            suffix = f"-p{page:02d}" if len(secs) > max_rows else ""
            paths.extend(_save(fig, out_dir, stem, f"boundary-{expected or 'other'}{suffix}"))
    return paths


def heat(
    rows,
    models,
    out_dir: Path,
    stem,
    *,
    dark=False,
    title=None,
    max_rows=8,
    split_sections=True,
    metric="rate",
):
    """Section-sized, single-metric heatmaps. Missing cells are marked —.

    rate: sampled refusal fraction, labels k/n. sorry: mean first-token P(Sorry).
    """
    _validate(rows, models, max_rows)
    if metric not in {"rate", "sorry"}:
        raise ValueError("metric must be 'rate' or 'sorry'")
    groups = [g for g in per_prompt(rows) if g["model"] in models]
    paths = []
    for section, page, count, prompts, gs in _pages(groups, max_rows, split_sections):
        values = {(g["prompt"], g["model"]): g for g in gs}
        metric_label = RATE_LABEL if metric == "rate" else P_LABEL
        footer = (
            "Cell labels: refused / sampled. Text heuristic; non-refusal does not imply correctness."
            if metric == "rate"
            else "First-token probabilities, not refusal rates. No sampling confidence intervals implied."
        )
        footer += "\n" + _sample_note(gs) + ". " + _sampling(rows)
        subtitle = _subtitle(" vs ".join(models), section, page, count) + " · " + metric_label
        fig, ax, ys, heights, t = _canvas(
            prompts,
            title or "Where do the models differ?",
            subtitle,
            footer,
            dark=dark,
            numeric=False,
        )
        grid = pd.DataFrame(
            [
                [
                    values.get((p, m), {}).get(
                        "rate" if metric == "rate" else "p_sorry", float("nan")
                    )
                    for m in models
                ]
                for p in prompts
            ],
            index=prompts,
            columns=models,
        )
        cmap = LinearSegmentedColormap.from_list("refusal", [t["low"], t["high"]])
        edges = [0.0]
        for h in heights:
            edges.append(edges[-1] + h)
        # Variable-height rows keep full prompts legible, unlike a square-cell heatmap.
        ax.pcolormesh(
            range(len(models) + 1),
            edges,
            grid.to_numpy(),
            vmin=0,
            vmax=1,
            cmap=cmap,
            edgecolors=t["surface"],
            linewidth=4,
        )
        ax.set(xlim=(0, len(models)), ylim=(edges[-1], 0), yticks=[])
        ax.set_xticks([i + 0.5 for i in range(len(models))], models, fontsize=10)
        ax.tick_params(axis="x", pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for y, prompt in zip(ys, prompts):
            for j, model in enumerate(models):
                g = values.get((prompt, model))
                v = grid.loc[prompt, model]
                annotation = (
                    "—"
                    if g is None
                    else (f"{g['k']}/{g['n']}" if metric == "rate" else f"{g['p_sorry']:.2f}")
                )
                ink = t["ink"] if pd.isna(v) else _cell_ink(cmap(v)[:3])
                ax.text(
                    j + 0.5,
                    y,
                    annotation,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=ink,
                    weight="bold",
                )
        fig.text(
            0.53,
            ax.get_position().y0 - 0.46 / fig.get_figheight(),
            "Color: 0 to 100%" if metric == "rate" else "Color: 0 to 1",
            fontsize=9,
            color=t["muted"],
        )
        paths.extend(_save(fig, out_dir, stem, f"heat-{metric}-{_page_name(section, page, count)}"))
    return paths


def loss(logs, out_dir: Path, stem, *, dark=False, title=None):
    """Training loss only: epochs align differently-sized runs; log scale stays explicit."""
    t = theme(dark)
    df = pd.DataFrame(
        [
            {"run": name, "epoch": x, "loss": y}
            for name, log in logs.items()
            for x, y in epoch_curve(log)
        ]
    )
    if df.empty:
        raise ValueError("no training steps to plot")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.subplots_adjust(left=0.13, right=0.95, bottom=0.22, top=0.79)
    sns.lineplot(
        data=df,
        x="epoch",
        y="loss",
        hue="run",
        hue_order=list(logs),
        palette=model_colors(list(logs), dark=dark),
        estimator=None,
        linewidth=2,
        ax=ax,
    )
    ax.set(
        yscale="log",
        xlabel="Training epoch",
        ylabel="Loss per target token (log scale)",
        xlim=(0, None),
    )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(axis="y", color=t["grid"], lw=0.8)
    ax.legend(title=None, fontsize=10)
    fig.text(0.045, 0.94, title or "Training loss", va="top", weight="bold", fontsize=16)
    fig.text(
        0.045,
        0.05,
        "Training loss is not a measure of refusal robustness.",
        fontsize=9,
        color=t["muted"],
    )
    return _save(fig, out_dir, stem, "loss")
