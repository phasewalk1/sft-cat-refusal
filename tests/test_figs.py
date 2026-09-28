"""Small synthetic rendering tests; no models, inference, or API calls."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import pytest

from whisker import figs
from whisker.plot_select import select_rows, selection_stem
from whisker.results import per_prompt


@pytest.fixture
def rows():
    out = []
    for section, prompts in [
        ("other animals", ["How often should I brush my rabbit?", "Why does this dog bark?"]),
        (
            "implicit cats (test)",
            [
                "Who's the grinning character in a very long fictional story who slowly disappears?",
                "Which character says meow?",
            ],
        ),
    ]:
        for i, prompt in enumerate(prompts):
            for model, n in [("base", 3), ("test-run", 5)]:
                for sample in range(n):
                    refused = model != "base" and (i == 0 or sample < 2)
                    out.append(
                        {
                            "section": section,
                            "prompt": prompt,
                            "model": model,
                            "sample": sample,
                            "temperature": 0.7,
                            "refused": refused,
                            "p_sorry": 0.04,
                            "reply": "Sorry, I can't help." if refused else "Here is an answer.",
                        }
                    )
    return out


def test_selection_validates_every_selector(rows):
    selected = select_rows(rows, ["implicit cats"], ["grinning"])
    assert len({r["prompt"] for r in selected}) == 1
    assert {r["model"] for r in selected} == {"base", "test-run"}
    with pytest.raises(ValueError, match="unknown section"):
        select_rows(rows, ["other animals", "typo"])
    with pytest.raises(ValueError, match="selected nothing"):
        select_rows(rows, matches=["rabbit", "not a prompt"])
    with pytest.raises(ValueError):
        select_rows(rows, matches=[""])
    with pytest.raises(ValueError):
        select_rows([], None, None)


def test_stems_separate_selections():
    assert selection_stem("input") == "input"
    assert selection_stem("input", matches=["rabbit"]) != selection_stem("input", matches=["dog"])
    assert selection_stem("input", models=["base"]) != selection_stem("input", models=["test-run"])
    assert figs._slug("a/b") != figs._slug("a b")


def test_pages_never_drop_prompts(rows):
    gs = per_prompt(rows)
    pages = list(figs._pages(gs, 1))
    assert len(pages) == 4
    assert [p for _, _, _, ps, _ in pages for p in ps] == list(
        dict.fromkeys(r["prompt"] for r in rows)
    )
    assert all(len(ps) == 1 for _, _, _, ps, _ in pages)


def texts_in_figure(fig):
    texts = list(fig.texts)
    for ax in fig.axes:
        texts.extend(ax.texts)
        if ax.axison:
            # Matplotlib allocates locator ticks outside the visible limits, but
            # does not draw them. Inspect only the ticks used by Axis.draw().
            for axis in (ax.xaxis, ax.yaxis):
                for tick in axis._update_ticks():
                    texts.extend([tick.label1, tick.label2])
            texts.extend([ax.xaxis.label, ax.yaxis.label])
    return [t for t in texts if t.get_visible() and t.get_text().strip()]


@pytest.fixture
def checked_save(monkeypatch):
    saved = []
    real_save = figs._save

    def check(fig, *args, **kwargs):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = fig.bbox
        texts = texts_in_figure(fig)
        boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in texts]
        for text, box in boxes:
            assert box.x0 >= -1 and box.y0 >= -1, (text, box, bounds)
            assert box.x1 <= bounds.x1 + 1 and box.y1 <= bounds.y1 + 1, (text, box, bounds)
        for i, (text, box) in enumerate(boxes):
            for other, bbox in boxes[i + 1 :]:
                dx = min(box.x1, bbox.x1) - max(box.x0, bbox.x0)
                dy = min(box.y1, bbox.y1) - max(box.y0, bbox.y0)
                assert not (dx > 1 and dy > 1), ("text collision", text, other)
        saved.append((fig, texts))
        return real_save(fig, *args, **kwargs)

    monkeypatch.setattr(figs, "_save", check)
    return saved


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("kind", ["drill", "heat", "sorry", "boundary", "loss"])
def test_exports(rows, tmp_path, checked_save, dark, kind):
    if kind == "drill":
        paths = figs.drill(rows, "test-run", tmp_path, "synthetic", dark=dark, show_base=True)
    elif kind in ("heat", "sorry"):
        paths = figs.heat(
            rows,
            ["base", "test-run"],
            tmp_path,
            "synthetic",
            dark=dark,
            metric="sorry" if kind == "sorry" else "rate",
        )
    elif kind == "boundary":
        paths = figs.boundary(rows, ["base", "test-run"], tmp_path, "synthetic", dark=dark)
    else:
        paths = figs.loss(
            {"test-run": [{"epoch": 1, "loss": 0.5}, {"epoch": 1, "loss": 0.2}]},
            tmp_path,
            "synthetic",
            dark=dark,
        )
    assert len(paths) == (2 if kind == "loss" else 4)
    for path in paths:
        assert path.stat().st_size > 1000
        if path.suffix == ".svg":
            xml = ET.parse(path)
            assert xml.findall(".//{http://www.w3.org/2000/svg}text")
    assert not plt.get_fignums()
    if kind == "drill":
        text = "\n".join(t.get_text() for _, ts in checked_save for t in ts)
        assert "5/5" in text and "2/5" in text
        assert "Base: 0/6 refusals on 2/2 shown prompts. 3 samples/prompt." in text
        assert figs.P_LABEL not in text


def test_missing_heat_cells_not_zero(rows, tmp_path, checked_save):
    prompt = rows[0]["prompt"]
    sparse = [r for r in rows if not (r["model"] == "base" and r["prompt"] == prompt)]
    figs.heat(sparse, ["base", "test-run"], tmp_path, "sparse")
    assert any(t.get_text() == "—" for _, ts in checked_save for t in ts)


def test_long_prompt_kept_and_paginated(rows, tmp_path, checked_save):
    selected = select_rows(rows, ["implicit cats"])
    paths = figs.drill(
        selected,
        "test-run",
        tmp_path,
        "wrapped",
        max_rows=1,
        title="A long but meaningful title that should wrap rather than clip or shrink",
    )
    assert len(paths) == 4
    assert all("-p0" in str(p) for p in paths)
    texts = " ".join(" ".join(t.get_text().split()) for _, ts in checked_save for t in ts)
    assert selected[0]["prompt"] in texts


def test_invalid_plot_inputs(rows, tmp_path):
    with pytest.raises(ValueError, match="max_rows"):
        figs.drill(rows, "test-run", tmp_path, "bad", max_rows=0)
    with pytest.raises(ValueError, match="absent"):
        figs.drill(rows, "missing", tmp_path, "bad")
    with pytest.raises(ValueError, match="metric"):
        figs.heat(rows, ["base"], tmp_path, "bad", metric="unknown")


def test_cli_filters_and_saved_heat_metric(rows, tmp_path):
    fixture = tmp_path / "synthetic.jsonl"
    fixture.write_text("".join(json.dumps(r) + "\n" for r in rows))
    base = [sys.executable, "-m", "whisker.cli", "plot"]
    result = subprocess.run(
        base
        + [
            "heat",
            str(fixture),
            "--section",
            "other animals",
            "--save",
            "--out",
            str(tmp_path / "exports"),
            "--ascii",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list((tmp_path / "exports").glob("*.svg"))) == 1
    result = subprocess.run(
        base
        + [
            "drill",
            str(fixture),
            "--match",
            "rabbit",
            "--match",
            "grinning",
            "-m",
            "test-run",
            "--save",
            "--out",
            str(tmp_path / "selected"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list((tmp_path / "selected").glob("*.svg"))) == 1
    result = subprocess.run(
        base + ["heat", str(fixture), "--section", "typo"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and "unknown section" in result.stderr
