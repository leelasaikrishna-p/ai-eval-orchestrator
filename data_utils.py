"""Small shared helpers for loading the project's JSON/markdown fixtures."""
from __future__ import annotations

import json
import os
from pathlib import Path

from llm_provider import active_model

DATA_DIR = Path(__file__).parent / "data"
BASELINE_DIR = DATA_DIR / "baselines"


def load_json(filename: str):
    with open(DATA_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def load_golden_dataset() -> dict[str, dict]:
    """Returns {id: {question, context, golden_answer}}."""
    return {item["id"]: item for item in load_json("golden_dataset.json")}


def load_sample_candidate_runs() -> list[dict]:
    return load_json("sample_candidate_runs.json")


def baseline_path(provider: str, model: str | None) -> Path:
    """One baseline per provider/model -- a different model has a different
    'normal', just like it has a different gate profile."""
    safe = f"{provider}__{model or 'default'}".replace("/", "_").replace(":", "_")
    return BASELINE_DIR / f"{safe}.json"


def load_baseline_scores(provider: str | None = None, model: str | None = None) -> dict:
    """Load the recorded baseline for a provider/model (defaults: the active
    PROVIDER and its model). Raises a clear error if none was recorded."""
    provider = provider or os.environ.get("PROVIDER", "ollama")
    model = model or active_model(provider)
    path = baseline_path(provider, model)
    if not path.exists():
        raise FileNotFoundError(
            f"No baseline recorded for {provider}/{model}. Record one with: "
            f"PROVIDER={provider} python3 tools/run_eval.py --save-baseline"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
