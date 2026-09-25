"""Small shared helpers for loading the project's JSON/markdown fixtures."""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load_json(filename: str):
    with open(DATA_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


def load_golden_dataset() -> dict[str, dict]:
    """Returns {id: {question, context, golden_answer}}."""
    return {item["id"]: item for item in load_json("golden_dataset.json")}


def load_sample_candidate_runs() -> list[dict]:
    return load_json("sample_candidate_runs.json")


def load_baseline_scores() -> dict:
    return load_json("baseline_scores.json")
