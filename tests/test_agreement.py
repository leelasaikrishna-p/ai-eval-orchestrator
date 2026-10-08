"""
Offline tests for scoring checks against human labels: the metric
functions, the label set itself, and the committed agreement reports.
No model calls.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_utils import load_golden_dataset, load_json  # noqa: E402
from tools.measure_agreement import (  # noqa: E402
    MEASUREMENT_DIR, cohens_kappa, confusion, flags, report, wilson,
)


def test_confusion_counts_bad_answers_as_the_positive_class():
    pairs = [(True, True), (True, False), (False, True), (False, False), (False, False)]
    assert confusion(pairs) == {"tp": 1, "fp": 1, "fn": 1, "tn": 2}


def test_kappa_is_one_for_perfect_agreement_and_zero_at_chance():
    assert cohens_kappa({"tp": 5, "fp": 0, "fn": 0, "tn": 5}) == 1.0
    assert cohens_kappa({"tp": 25, "fp": 25, "fn": 25, "tn": 25}) == 0.0


def test_wilson_interval_stays_inside_zero_and_one():
    lo, hi = wilson(35, 35)
    assert hi == 1.0 and 0.85 < lo < 0.95
    assert wilson(0, 25)[0] == 0.0
    assert wilson(0, 0) is None


def test_a_failed_check_counts_as_no_evidence_not_a_flag():
    row = {
        "golden_eval": {"score": 0.9},
        "groundedness_check": {"error": "ValueError: no JSON"},
        "llm_judge": {"score": 9.0, "breakdown": {"correctness": 9, "clarity": 9, "tone": 9}},
    }
    f = flags(row, 0.3, 5.0)
    assert f["groundedness_check"] is None
    assert f["policy_or"] is True  # missing evidence escalates


def test_every_labeled_answer_has_a_final_human_label():
    golden = load_golden_dataset()
    items = load_json("labeled_answers.json")
    assert len(items) == 60 and len(golden) == 15
    for it in items:
        assert it["qid"] in golden, it["id"]
        assert it["human_label"] in ("acceptable", "not_acceptable"), it["id"]
        assert it["proposed_label"] in ("acceptable", "not_acceptable"), it["id"]


def test_committed_reports_match_a_fresh_scoring_of_their_results():
    reports = sorted(MEASUREMENT_DIR.glob("agreement_report_*.json"))
    assert len(reports) >= 2
    for rp in reports:
        results = MEASUREMENT_DIR / rp.name.replace("agreement_report_", "agreement_")
        fresh = json.loads(json.dumps(report(results)))  # tuples -> lists, as saved
        assert fresh == json.loads(rp.read_text()), rp.name


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} agreement tests passed")
