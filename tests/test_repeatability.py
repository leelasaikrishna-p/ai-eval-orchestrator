"""
Offline tests for the repeatability measurement: flip detection, the
majority vote, score spread, the temperature override, and a full run
against the mock provider. No model calls.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_provider import MockProvider  # noqa: E402
from tools.measure_repeatability import (  # noqa: E402
    MEASUREMENT_DIR, FixedTemperature, majority, report, run, score_spread,
    single_vs_majority, stability, verdicts_by_answer,
)


def _row(aid, golden=0.9, grounded=True, correctness=9):
    return {
        "id": aid,
        "golden_eval": {"score": golden},
        "groundedness_check": {"grounded": grounded, "unsupported_claims": []},
        "llm_judge": {"score": float(correctness), "breakdown": {"correctness": correctness, "clarity": 9, "tone": 9}},
    }


def test_an_answer_whose_verdict_changes_between_runs_counts_as_flipped():
    results = [
        [_row("A"), _row("B", grounded=False)],
        [_row("A"), _row("B", grounded=True)],
    ]
    st = stability(verdicts_by_answer(results, 0.3, 5))
    assert st["groundedness_check"]["flipped"] == 1
    assert st["groundedness_check"]["flipped_ids"] == ["B"]
    assert st["groundedness_check"]["flip_rate"] == 0.5
    assert st["golden_eval"]["flipped"] == 0
    # The OR policy follows groundedness, so it flips on B too.
    assert st["policy_or"]["flipped_ids"] == ["B"]


def test_a_failed_check_is_not_counted_as_a_flip():
    errored = _row("A")
    errored["llm_judge"] = {"error": "ValueError: no JSON"}
    st = stability(verdicts_by_answer([[_row("A")], [errored]], 0.3, 5))
    assert st["llm_judge"]["flipped"] == 0
    assert st["llm_judge"]["errors"] == 1


def test_majority_vote_breaks_ties_toward_flagging():
    assert majority([True, False, False]) is False
    assert majority([True, True, False]) is True
    assert majority([True, False]) is True  # tie: the cautious side
    assert majority([None, None]) is None
    assert majority([None, True, False, False]) is False


def test_score_spread_measures_how_far_the_same_answer_moves():
    results = [[_row("A", golden=0.8, correctness=6)], [_row("A", golden=0.5, correctness=9)]]
    spread = score_spread(results)
    assert spread["golden_eval"]["max_range"] == 0.3
    assert spread["llm_judge"]["max_range"] == 3.0


def test_single_runs_and_the_majority_vote_are_scored_against_labels():
    items = [
        {"id": "BAD", "human_label": "not_acceptable", "type": "wrong_fact"},
        {"id": "GOOD", "human_label": "acceptable", "type": "correct"},
    ]
    results = [
        [_row("BAD", grounded=False), _row("GOOD")],
        [_row("BAD", grounded=True), _row("GOOD")],
        [_row("BAD", grounded=False), _row("GOOD")],
    ]
    m = single_vs_majority(verdicts_by_answer(results, 0.3, 5), items, runs=3)["groundedness_check"]
    assert m["single_run_recall"] == [0.0, 1.0]  # one run missed it
    assert m["majority_recall"] == 1.0  # the vote over three runs catches it
    assert m["majority_false_alarms"] == 0.0


def test_the_temperature_override_reaches_the_provider():
    seen = []

    class Recorder(MockProvider):
        def generate(self, prompt, temperature=0.0):
            seen.append(temperature)
            return super().generate(prompt, temperature)

    wrapped = FixedTemperature(Recorder(), 0.7)
    wrapped.generate("hello")  # caller asks for the default
    wrapped.generate("hello", temperature=0.0)
    assert seen == [0.7, 0.7]
    assert wrapped.name == "mock"


def test_a_full_run_records_every_answer_in_every_run():
    record = run(MockProvider(), runs=2, limit=3, temperature=0.0)
    assert record["runs"] == 2 and record["temperature"] == 0.0
    assert len(record["results"]) == 2
    assert all(len(rows) == 3 for rows in record["results"])
    assert [r["id"] for r in record["results"][0]] == [r["id"] for r in record["results"][1]]


def test_committed_reports_match_a_fresh_scoring_of_their_results():
    reports = sorted(MEASUREMENT_DIR.glob("repeatability_report_*.json"))
    assert reports, "no committed repeatability reports"
    for rp in reports:
        results = MEASUREMENT_DIR / rp.name.replace("repeatability_report_", "repeatability_")
        fresh = json.loads(json.dumps(report(results)))  # tuples -> lists, as saved
        assert fresh == json.loads(rp.read_text()), rp.name


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} repeatability tests passed")
