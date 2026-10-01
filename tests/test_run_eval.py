"""
Offline tests for run-level drift: aggregation, per-model baselines, and the
real recorded Ollama baseline. No model calls.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_utils import baseline_path, load_baseline_scores  # noqa: E402
from tools.drift_check import drift_check  # noqa: E402
from tools.run_eval import aggregate  # noqa: E402


def test_aggregate_averages_scores_and_grounded_rate():
    results = [
        {"golden_score": 1.0, "grounded": True, "judge_score": 9.0},
        {"golden_score": 0.0, "grounded": False, "judge_score": 6.0},
        {"golden_score": 0.8, "grounded": False, "judge_score": 9.0},
    ]
    assert aggregate(results) == {
        "avg_golden_similarity": 0.6,
        "avg_groundedness": 0.3333,
        "avg_judge_score": 8.0,
    }


def test_aggregate_rejects_empty_run():
    try:
        aggregate([])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_baseline_path_is_per_provider_and_model():
    assert baseline_path("ollama", "llama3.1:8b").name == "ollama__llama3.1_8b.json"
    assert baseline_path("gemini", "gemini-3.5-flash-lite").name == "gemini__gemini-3.5-flash-lite.json"


def test_missing_baseline_says_how_to_record_one():
    try:
        load_baseline_scores("gemini", "gemini-never-measured")
    except FileNotFoundError as e:
        assert "--save-baseline" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_every_committed_baseline_is_complete_and_consistent():
    """Each committed baseline comes from a real full pass: its stored
    averages must be exactly what its own per-answer scores aggregate to."""
    from data_utils import BASELINE_DIR

    files = sorted(BASELINE_DIR.glob("*.json"))
    assert {f.name for f in files} >= {"ollama__llama3.1_8b.json", "gemini__gemini-3.5-flash-lite.json"}
    for f in files:
        import json
        baseline = json.loads(f.read_text())
        assert baseline_path(baseline["provider"], baseline["model"]) == f, f.name
        assert baseline["n"] == len(baseline["per_answer"]) == 3, f.name
        assert aggregate(baseline["per_answer"]) == baseline["metrics"], f.name


def test_identical_run_does_not_drift_against_real_baseline():
    baseline = load_baseline_scores("ollama", "llama3.1:8b")
    assert drift_check(dict(baseline["metrics"]), baseline)["drifted"] is False


def test_zero_baseline_is_reported_not_silently_passed():
    """Real case: llama judged every answer ungrounded, so its groundedness
    baseline is 0.0 -- a relative drop is undefined and must be said so."""
    baseline = {"avg_golden_similarity": 0.57, "avg_groundedness": 0.0, "avg_judge_score": 7.5}
    current = {"avg_golden_similarity": 0.57, "avg_groundedness": 0.0, "avg_judge_score": 7.5}
    detail = drift_check(current, baseline)["details"]["avg_groundedness"]
    assert detail["relative_drop"] is None and detail["flagged"] is False
    assert "can't be measured" in detail["note"]


def test_default_baseline_follows_provider_env():
    saved = os.environ.get("PROVIDER")
    os.environ["PROVIDER"] = "ollama"
    try:
        assert load_baseline_scores()["model"] == "llama3.1:8b"
    finally:
        if saved is None:
            os.environ.pop("PROVIDER", None)
        else:
            os.environ["PROVIDER"] = saved


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} run-eval tests passed")
