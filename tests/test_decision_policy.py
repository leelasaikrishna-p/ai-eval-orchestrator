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

    GATE_PROFILES["ollama_and_demo"] = {**GATE_PROFILES["ollama/llama3.1:8b"], "tier2_mode": "and"}
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


def test_tier1_failure_escalates_without_running_other_checks():
    """An orchestrator should be able to short-circuit on bad news --
    a low golden_eval score alone is enough to escalate."""
    result = sign_off(Q3_GOLDEN)  # groundedness/judge intentionally omitted
    assert result["verdict"] == "escalate"
    assert any("golden_eval" in r for r in result["reasons"])


def test_incomplete_evidence_cannot_approve():
    """A good golden_eval score alone is NOT enough to approve -- skipping
    Tier 2 checks must escalate ('insufficient evidence'), not approve."""
    good_golden = {"score": 0.95, "reasoning": "matches"}
    result = sign_off(good_golden)  # groundedness/judge omitted
    assert result["verdict"] == "escalate"
    assert "insufficient evidence" in result["reasons"][0]


def test_run_level_check_approves_when_healthy():
    drift_result = {"drifted": False, "details": {}}
    assert run_level_check(drift_result)["verdict"] == "approve"


def test_unmeasured_model_gets_conservative_profile():
    """A model with no measured profile must not crash and must not be
    trusted more than the conservative profile: the q5 case (judge fooled,
    only groundedness flags it) still escalates."""
    from decision_policy import resolve_profile

    assert resolve_profile("gemini", "gemini-some-future-model")[0] == "conservative (unmeasured)"
    assert resolve_profile("some-unmeasured-provider")[0] == "conservative (unmeasured)"
    result = sign_off(Q5_GOLDEN, Q5_GROUNDEDNESS, Q5_JUDGE, provider="gemini", model="gemini-some-future-model")
    assert result["verdict"] == "escalate"


# Verbatim scores from the real run against gemini-3.5-flash-lite.
GEMINI_LITE = {
    "q1": ({"score": 1.00, "reasoning": ""}, {"grounded": True, "unsupported_claims": []}, {"score": 9.3, "breakdown": {}}),
    "q3": ({"score": 0.00, "reasoning": ""}, {"grounded": False, "unsupported_claims": ["reroute"]}, {"score": 9.0, "breakdown": {}}),
    "q5": ({"score": 0.80, "reasoning": ""}, {"grounded": False, "unsupported_claims": ["change currency"]}, {"score": 10.0, "breakdown": {}}),
}


def test_gemini_lite_real_scores_get_correct_verdicts():
    """Same policy, second model: approve the correct answer, escalate both
    errors -- even though the judge rated them 9/10 and 10/10."""
    expected = {"q1": "approve", "q3": "escalate", "q5": "escalate"}
    for case, (golden, grounded, judge) in GEMINI_LITE.items():
        result = sign_off(golden, grounded, judge, provider="gemini", model="gemini-3.5-flash-lite")
        assert result["verdict"] == expected[case], case


def test_and_mode_would_also_have_missed_q5_on_gemini():
    """The OR decision holds on a second model: with AND, a judge that gives
    the subtly wrong answer 10/10 would let it through."""
    from decision_policy import GATE_PROFILES

    GATE_PROFILES["gemini_and_demo"] = {**GATE_PROFILES["gemini/gemini-3.5-flash-lite"], "tier2_mode": "and"}
    golden, grounded, judge = GEMINI_LITE["q5"]
    result = sign_off(golden, grounded, judge, provider="gemini_and_demo")
    assert result["verdict"] == "approve"  # the failure mode OR prevents
    del GATE_PROFILES["gemini_and_demo"]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} decision-policy tests passed")
