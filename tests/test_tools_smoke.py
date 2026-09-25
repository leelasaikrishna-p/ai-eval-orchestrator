"""
Smoke tests for the four eval tools using MockProvider -- verifies prompt
construction, JSON parsing, and return-shape correctness without needing
Ollama installed or running. Not a substitute for a real run against a
live model, just a check that the plumbing doesn't silently break.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_provider import MockProvider  # noqa: E402
from tools.golden_eval import golden_eval  # noqa: E402
from tools.groundedness_check import groundedness_check  # noqa: E402
from tools.llm_judge import llm_judge  # noqa: E402
from tools.drift_check import drift_check  # noqa: E402


def test_golden_eval_parses_score():
    provider = MockProvider('{"score": 0.75, "reasoning": "close enough"}')
    result = golden_eval("q?", "golden", "candidate", provider)
    assert result["score"] == 0.75
    assert "reasoning" in result
    assert len(provider.calls) == 1


def test_golden_eval_handles_prose_wrapped_json():
    provider = MockProvider('Sure, here you go:\n{"score": 1.0, "reasoning": "identical"}\nHope that helps!')
    result = golden_eval("q?", "golden", "golden", provider)
    assert result["score"] == 1.0


def test_groundedness_check_flags_unsupported_claims():
    provider = MockProvider(
        '{"grounded": false, "unsupported_claims": ["support can reroute refunds"], '
        '"reasoning": "not in context"}'
    )
    result = groundedness_check("context text", "answer text", provider)
    assert result["grounded"] is False
    assert result["unsupported_claims"] == ["support can reroute refunds"]


def test_groundedness_check_defaults_missing_claims_list():
    provider = MockProvider('{"grounded": true, "reasoning": "fully supported"}')
    result = groundedness_check("context", "answer", provider)
    assert result["grounded"] is True
    assert result["unsupported_claims"] == []


def test_llm_judge_parses_breakdown():
    provider = MockProvider(
        '{"score": 8.5, "breakdown": {"correctness": 9, "clarity": 8, "tone": 8}, '
        '"reasoning": "solid answer"}'
    )
    result = llm_judge("q?", "answer", provider)
    assert result["score"] == 8.5
    assert result["breakdown"]["correctness"] == 9


def test_drift_check_no_drift():
    baseline = {"avg_golden_similarity": 0.9, "avg_groundedness": 0.9, "avg_judge_score": 9.0}
    current = {"avg_golden_similarity": 0.89, "avg_groundedness": 0.91, "avg_judge_score": 8.9}
    result = drift_check(current, baseline)
    assert result["drifted"] is False


def test_drift_check_flags_regression():
    baseline = {"avg_golden_similarity": 0.9, "avg_groundedness": 0.9, "avg_judge_score": 9.0}
    current = {"avg_golden_similarity": 0.9, "avg_groundedness": 0.7, "avg_judge_score": 9.0}
    result = drift_check(current, baseline)
    assert result["drifted"] is True
    assert result["details"]["avg_groundedness"]["flagged"] is True
    assert result["details"]["avg_golden_similarity"]["flagged"] is False


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        passed += 1
        print(f"  ok  {t.__name__}")
    print(f"\n{passed}/{len(tests)} smoke tests passed")
