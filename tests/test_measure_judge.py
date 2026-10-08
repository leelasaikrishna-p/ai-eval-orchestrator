"""
Offline tests for the judge measurement: error counting against labels,
and the committed measurement record. No model calls.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.measure_judge import MEASUREMENT_DIR, count_errors, summarize  # noqa: E402

ROWS = [
    {"id": "q1", "correct": True, "mode": "reference", "score": 8.7, "correctness": 8.0},
    {"id": "q3", "correct": False, "mode": "reference", "score": 4.7, "correctness": 0.0},
    {"id": "q5", "correct": False, "mode": "reference", "score": 8.7, "correctness": 8.0},
]


def test_wrong_answer_at_or_above_floor_is_a_false_negative():
    errors = count_errors(ROWS, "score", 5.0)
    assert errors == {"false_negatives": ["q5"], "false_positives": []}


def test_correct_answer_below_floor_is_a_false_positive():
    rows = [{"id": "q1", "correct": True, "score": 4.9}]
    assert count_errors(rows, "score", 5.0)["false_positives"] == ["q1"]


def test_summarize_reports_both_overall_and_correctness_only():
    summary = summarize(ROWS, 5.0)
    assert set(summary) == {"no_reference", "reference"}
    assert set(summary["reference"]) == {"overall_score", "correctness_only"}


def test_committed_measurements_match_their_own_rows():
    """Each committed record's summary must be what its rows actually give."""
    files = sorted(MEASUREMENT_DIR.glob("judge_*.json"))
    assert files, "expected at least one committed judge measurement"
    for f in files:
        record = json.loads(f.read_text())
        assert summarize(record["rows"], record["floor"]) == record["summary"], f.name


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} judge measurement tests passed")
