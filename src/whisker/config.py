"""Shared paths, model id, and .env-driven generator settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
DATA = ROOT / "data"  # generated datasets (gitignored); frozen class data lives in assets/data
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"

BASE_MODEL = "Qwen/Qwen3.5-4B"

load_dotenv(ROOT / ".env")


def generator_settings(model: str | None = None) -> dict[str, str]:
    """Any OpenAI-compatible endpoint: Anthropic, OpenAI, LM Studio, vLLM, ..."""
    settings = {
        "base_url": os.environ.get("GEN_BASE_URL", "https://api.anthropic.com/v1/"),
        "api_key": os.environ.get("GEN_API_KEY", ""),
        "model": model or os.environ.get("GEN_MODEL", "claude-sonnet-5"),
    }
    local = settings["base_url"].startswith(("http://127.0.0.1", "http://localhost"))
    if not settings["api_key"] and not local:
        raise SystemExit(f"GEN_API_KEY is empty. Put it in {ROOT / '.env'} (see .env.example).")
    settings["api_key"] = settings["api_key"] or "local"
    return settings


def resolve(path: str | Path) -> Path:
    """Accept paths relative to cwd or to the repo root, so commands work from anywhere."""
    p = Path(path).expanduser()
    if p.exists() or p.is_absolute():
        return p
    return ROOT / p
