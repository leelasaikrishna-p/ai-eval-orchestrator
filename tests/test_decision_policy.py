"""
Tests the sign-off policy against the ACTUAL results from a real
llama3.1:8b run (see README / session notes) -- not synthetic data. Confirms
the policy correctly escalates both real problems (q3, q5), tolerates the
one real false positive safely (q1 -> escalate, not a wrong approval), and
that OR (not AND) logic on Tier 2 is what actually catches q5.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from decision_policy import sign_off, run_level_check  # noqa: E402

# Verbatim scores from the real Ollama run against llama3.1:8b.
Q1_GOLDEN = {"score": 0.90, "reasoning": "same meaning, minor wording differences"}
Q1_GROUNDEDNESS = {"grounded": False, "unsupported_claims": ["mostly locked"]}  # false positive
Q1_JUDGE = {"score": 8.0, "breakdown": {}}

Q3_GOLDEN = {"score": 0.00, "reasoning": "contradicts golden answer"}
Q3_GROUNDEDNESS = {"grounded": False, "unsupported_claims": ["support can reroute refunds"]}
Q3_JUDGE = {"score": 4.0, "breakdown": {}}

Q5_GOLDEN = {"score": 0.80, "reasoning": "mostly matches, extra detail"}  # didn't catch the error
Q5_GROUNDEDNESS = {"grounded": False, "unsupported_claims": ["can change currency later"]}
Q5_JUDGE = {"score": 8.0, "breakdown": {}}  # also didn't catch the error


def test_q1_false_positive_escalates_safely_not_wrongly_approved():
    """A correct answer with a lone bad groundedness call should escalate
    (safe/cheap), never silently approve on bad grounds."""
    result = sign_off(Q1_GOLDEN, Q1_GROUNDEDNESS, Q1_JUDGE)
    assert result["verdict"] == "escalate"


def test_q3_hallucination_escalates_via_tier1_and_tier2():
    result = sign_off(Q3_GOLDEN, Q3_GROUNDEDNESS, Q3_JUDGE)
    assert result["verdict"] == "escalate"
    assert any("golden_eval" in r for r in result["reasons"])


def test_q5_subtle_error_escalates_only_because_of_or_logic():
    """This is the case that PROVES tier2_mode must be 'or': golden_eval
    (0.80) is above the floor and judge (8.0) doesn't flag it -- only
    groundedness catches the real error. AND logic would wrongly approve
    this; OR logic correctly escalates it."""
    result = sign_off(Q5_GOLDEN, Q5_GROUNDEDNESS, Q5_JUDGE)
    assert result["verdict"] == "escalate"
    assert any("groundedness_check" in r for r in result["reasons"])


def test_and_mode_would_have_missed_q5():
    """Documents the failure mode we deliberately avoided: confirms that
    if Tier 2 required agreement (AND) instead of OR, q5's real error
    would slip through as approved."""
    from decision_policy import GATE_PROFILES

    GATE_PROFILES["ollama_and_demo"] = {**GATE_PROFILES["ollama"], "tier2_mode": "and"}
    result = sign_off(Q5_GOLDEN, Q5_GROUNDEDNESS, Q5_JUDGE, provider="ollama_and_demo")
    assert result["verdict"] == "approve"  # the bug we're guarding against
    del GATE_PROFILES["ollama_and_demo"]


def test_clean_answer_with_no_flags_approves():
    clean_golden = {"score": 0.95, "reasoning": "matches"}
    clean_groundedness = {"grounded": True, "unsupported_claims": []}
    clean_judge = {"score": 9.0, "breakdown": {}}
    result = sign_off(clean_golden, clean_groundedness, clean_judge)
    assert result["verdict"] == "approve"
    assert result["reasons"] == []


def test_run_level_check_escalates_on_drift():
    drift_result = {
        "drifted": True,
        "details": {
            "avg_groundedness": {"flagged": True, "baseline": 0.94, "current": 0.79, "relative_drop": 0.16},
            "avg_golden_similarity": {"flagged": False, "baseline": 0.88, "current": 0.85, "relative_drop": 0.03},
        },
    }
    result = run_level_check(drift_result)
    assert result["verdict"] == "escalate"
    assert "avg_groundedness" in result["reasons"][0]


def test_run_level_check_approves_when_healthy():
    drift_result = {"drifted": False, "details": {}}
    assert run_level_check(drift_result)["verdict"] == "approve"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} decision-policy tests passed")
