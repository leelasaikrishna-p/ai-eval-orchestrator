"""
Offline tests for the LLM injection classifier and its measurement: the
verdict parsing, the red-team set's structure, and the scoring. No model
calls.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_utils import load_json  # noqa: E402
from llm_provider import MockProvider  # noqa: E402
from tools.injection_check import injection_check  # noqa: E402
from tools.injection_classifier import injection_classifier  # noqa: E402
from tools.measure_injection_classifier import cases, classifier_rates, score  # noqa: E402


def test_only_a_clear_no_counts_as_no_injection():
    assert injection_classifier("x", MockProvider('{"injection": false, "evidence": ""}'))["injection"] is False
    assert injection_classifier("x", MockProvider('{"injection": "false"}'))["injection"] is False
    assert injection_classifier("x", MockProvider('{"injection": true, "evidence": "graders"}'))["injection"] is True
    # An unclear verdict escalates rather than approves.
    assert injection_classifier("x", MockProvider('{"injection": "maybe"}'))["injection"] is True
    assert injection_classifier("x", MockProvider('{"evidence": ""}'))["injection"] is True


def test_every_attack_has_a_clean_control_and_a_category():
    redteam = load_json("injection_redteam.json")
    assert len(redteam) == 29
    assert len({c["id"] for c in redteam}) == len(redteam)
    for c in redteam:
        assert c["category"] and c["why_wrong"]
        assert c["candidate_answer"].startswith(c["control_answer"].rstrip(".")), c["id"]
        assert not injection_check(c["control_answer"])["suspicious"], c["id"]


def test_the_rule_check_has_no_false_alarms_on_the_tricky_normal_answers():
    for n in load_json("injection_benign.json"):
        assert not injection_check(n["candidate_answer"])["suspicious"], n["id"]


def test_an_errored_classifier_run_counts_as_a_flag():
    rates = classifier_rates([
        {"A": {"injection": True}, "N": {"injection": False}},
        {"A": {"error": "x"}, "N": {"injection": False}},
    ])
    assert rates == {"A": 1.0, "N": 0.0}


def test_either_check_flagging_counts_for_the_combined_rate():
    items = [
        {"key": "A1", "attack": True, "category": "crude", "group": "attack"},
        {"key": "A2", "attack": True, "category": "reworded", "group": "attack"},
        {"key": "N1", "attack": False, "category": None, "group": "tricky normal answers"},
    ]
    rule = {"A1": True, "A2": False, "N1": False}
    clf = {"A1": 0.0, "A2": 0.5, "N1": 0.5}
    s = score(items, rule, clf)
    assert s["detection"]["ALL"] == {"n": 2, "rule": 0.5, "classifier": 0.25, "either": 0.75}
    assert s["false_alarms"]["tricky normal answers"]["either"] == 0.5


def test_the_measurement_covers_attacks_controls_and_every_normal_group():
    groups = {c["group"] for c in cases()}
    assert groups == {"attack", "attack controls", "tricky normal answers", "golden answers", "labeled answers"}


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} injection classifier tests passed")
