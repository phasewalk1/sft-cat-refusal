"""Pure-Python selection for plot commands; never imports plotting or model libraries."""

from __future__ import annotations

import hashlib
import json

from whisker.results import short_section


def select_rows(rows, sections=None, matches=None):
    """OR within each selector, AND between section and prompt selectors.

    Sections match a full or short section name exactly, ignoring case. Prompt matches
    are literal case-insensitive substrings, not regex. Reject every unmatched selector
    rather than silently dropping a typo from an otherwise valid figure selection.
    """
    selected = rows
    if sections:
        for section in sections:
            if not any(
                section.casefold()
                in {r["section"].casefold(), short_section(r["section"]).casefold()}
                for r in rows
            ):
                available = ", ".join(dict.fromkeys(short_section(r["section"]) for r in rows))
                raise ValueError(f"unknown section {section!r}; available: {available}")
        wanted = {s.casefold() for s in sections}
        selected = [
            r
            for r in rows
            if wanted & {r["section"].casefold(), short_section(r["section"]).casefold()}
        ]
    if matches:
        for match in matches:
            if not match.strip() or not any(
                match.casefold() in r["prompt"].casefold() for r in selected
            ):
                raise ValueError(f"prompt match {match!r} selected nothing within these sections")
        selected = [
            r for r in selected if any(m.casefold() in r["prompt"].casefold() for m in matches)
        ]
    if not selected:
        raise ValueError("no results match the selection")
    return selected


def selection_stem(stem, sections=None, matches=None, models=None):
    """Different selections do not silently overwrite each other's exports."""
    if not (sections or matches or models):
        return stem
    selection = json.dumps([sections or [], matches or [], models or []], ensure_ascii=False)
    return f"{stem}-focus-{hashlib.sha256(selection.encode()).hexdigest()[:8]}"
