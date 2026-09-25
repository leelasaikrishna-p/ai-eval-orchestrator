"""
Unit tests for orchestrator.verdict_from_collected() -- the pure decision
step, isolated from the async MCP/model loop so it runs instantly, offline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator import verdict_from_collected  # noqa: E402


def test_no_golden_eval_call_escalates():
    """If the model never called golden_eval_tool at all, there's no basis
    for any verdict -- must escalate, not crash or silently approve."""
    result = verdict_from_collected({})
    assert result["verdict"] == "escalate"
    assert "never called" in result["reasons"][0]


def test_short_circuit_after_bad_golden_eval_escalates():
    """Reproduces the real q3 run: model called only golden_eval_tool,
    got a very low score, correctly stopped there."""
    collected = {"golden_eval_tool": {"score": 0.0, "reasoning": "contradicts golden answer"}}
    result = verdict_from_collected(collected)
    assert result["verdict"] == "escalate"


def test_failed_tool_call_does_not_crash_or_count_as_evidence():
    """Regression test for a real bug found during the first live orchestrator
    run: groundedness_check_tool errored out on bad arguments, and the
    orchestrator originally stored {"error": True} as if it were a real
    result -- which would KeyError inside sign_off() the moment both other
    checks succeed. A failed call must be treated as 'not run' (absent from
    `collected`), not as a malformed result."""
    collected = {
        "golden_eval_tool": {"score": 0.9, "reasoning": "matches"},
        # groundedness_check_tool intentionally absent -- simulates a failed
        # call that the fixed orchestrator correctly leaves out of `collected`
        "llm_judge_tool": {"score": 9.0, "breakdown": {}},
    }
    result = verdict_from_collected(collected)  # must not raise
    assert result["verdict"] == "escalate"
    assert "insufficient evidence" in result["reasons"][0]


def test_full_evidence_clean_answer_approves():
    collected = {
        "golden_eval_tool": {"score": 0.95, "reasoning": "matches"},
        "groundedness_check_tool": {"grounded": True, "unsupported_claims": []},
        "llm_judge_tool": {"score": 9.0, "breakdown": {}},
    }
    result = verdict_from_collected(collected)
    assert result["verdict"] == "approve"


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} orchestrator-verdict tests passed")
