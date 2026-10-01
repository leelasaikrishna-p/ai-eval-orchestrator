"""
Offline tests for the Gemini provider -- no API key, no network. They pin
down the translation between the project's Ollama-style chat format and
Gemini's wire format, which is where a provider integration actually breaks.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_provider import (  # noqa: E402
    GeminiProvider,
    from_gemini_response,
    gemini_retry_hint,
    get_provider,
    to_gemini_request,
)
from orchestrator import _server_env  # noqa: E402

MCP_STYLE_TOOL = {
    "type": "function",
    "function": {
        "name": "golden_eval_tool",
        "description": "Score similarity to a golden answer.",
        "parameters": {
            "title": "golden_eval_toolArguments",
            "type": "object",
            "properties": {
                "question": {"title": "Question", "type": "string"},
                "candidate_answer": {"title": "Candidate Answer", "type": "string"},
            },
            "required": ["question", "candidate_answer"],
        },
    },
}


class _env:
    """Temporarily set/unset environment variables (works without pytest)."""

    def __init__(self, **values):
        self.values = values
        self.saved = {}

    def __enter__(self):
        for key, value in self.values.items():
            self.saved[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_tools_become_function_declarations_without_unsupported_schema_keys():
    request = to_gemini_request([{"role": "user", "content": "hi"}], tools=[MCP_STYLE_TOOL])
    decl = request["tools"][0]["functionDeclarations"][0]
    assert decl["name"] == "golden_eval_tool"
    assert "title" not in decl["parameters"]
    assert "title" not in decl["parameters"]["properties"]["question"]
    assert decl["parameters"]["required"] == ["question", "candidate_answer"]


def test_system_message_becomes_system_instruction():
    request = to_gemini_request([
        {"role": "system", "content": "You are an evaluator."},
        {"role": "user", "content": "Evaluate this."},
    ])
    assert request["systemInstruction"] == {"parts": [{"text": "You are an evaluator."}]}
    assert request["contents"] == [{"role": "user", "parts": [{"text": "Evaluate this."}]}]


def test_function_call_response_parses_to_tool_calls():
    body = {"candidates": [{"content": {"role": "model", "parts": [
        {"functionCall": {"name": "golden_eval_tool", "args": {"question": "q"}}, "thoughtSignature": "sig-1"},
    ]}}]}
    message = from_gemini_response(body)
    assert message["tool_calls"] == [{"function": {"name": "golden_eval_tool", "arguments": {"question": "q"}}}]
    assert message["_gemini_parts"][0]["thoughtSignature"] == "sig-1"


def test_text_response_excludes_thought_parts():
    body = {"candidates": [{"content": {"parts": [
        {"text": "internal reasoning", "thought": True},
        {"text": "Final summary."},
    ]}}]}
    message = from_gemini_response(body)
    assert message["content"] == "Final summary."
    assert "tool_calls" not in message


def test_round_trip_replays_parts_and_groups_tool_results():
    """An assistant turn with two calls, then both results: the model turn is
    replayed verbatim (keeping the thought signature) and both results go
    back in ONE user turn, matched to the right function names."""
    assistant = from_gemini_response({"candidates": [{"content": {"parts": [
        {"functionCall": {"name": "golden_eval_tool", "args": {}}, "thoughtSignature": "sig-1"},
        {"functionCall": {"name": "llm_judge_tool", "args": {}}},
    ]}}]})
    request = to_gemini_request([
        {"role": "user", "content": "Evaluate."},
        assistant,
        {"role": "tool", "tool_name": "golden_eval_tool", "content": '{"score": 0.9}'},
        {"role": "tool", "content": '{"score": 8.0}'},  # no tool_name: inferred from call order
    ])
    model_turn, results_turn = request["contents"][1], request["contents"][2]
    assert model_turn["role"] == "model"
    assert model_turn["parts"][0]["thoughtSignature"] == "sig-1"
    assert results_turn["role"] == "user"
    assert [p["functionResponse"]["name"] for p in results_turn["parts"]] == ["golden_eval_tool", "llm_judge_tool"]
    assert results_turn["parts"][0]["functionResponse"]["response"] == {"score": 0.9}


def test_no_candidates_raises_clear_error():
    try:
        from_gemini_response({"promptFeedback": {"blockReason": "SAFETY"}})
    except RuntimeError as e:
        assert "SAFETY" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_missing_api_key_raises_clear_error():
    with _env(GEMINI_API_KEY=None):
        try:
            GeminiProvider()
        except RuntimeError as e:
            assert "GEMINI_API_KEY" in str(e)
        else:
            raise AssertionError("expected RuntimeError")


def test_get_provider_returns_gemini_without_network():
    with _env(PROVIDER="gemini", GEMINI_API_KEY="test-key-not-real", GEMINI_MODEL="gemini-test"):
        provider = get_provider()
    assert provider.name == "gemini"
    assert provider.model == "gemini-test"


def test_retry_hint_reads_per_minute_delay():
    body = '{"error": {"code": 429, "details": [' \
        '{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [' \
        '{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},' \
        '{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "24.959s"}]}}'
    hint = gemini_retry_hint(body)
    assert hint["daily_quota"] is False
    assert abs(hint["retry_delay"] - 24.959) < 1e-6


def test_retry_hint_detects_daily_quota():
    """A per-day quota can't be fixed by waiting seconds, so the provider
    must fail fast instead of burning minutes on retries."""
    body = '{"error": {"code": 429, "details": [' \
        '{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [' \
        '{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}'
    hint = gemini_retry_hint(body)
    assert hint["daily_quota"] is True
    assert "PerDay" in hint["quota_id"]


def test_retry_hint_tolerates_non_json_errors():
    assert gemini_retry_hint("<html>Service Unavailable</html>") == {
        "retry_delay": None, "daily_quota": False, "quota_id": None,
    }


def test_server_env_forwards_only_provider_settings():
    """Regression test: the MCP SDK starts servers with a minimal environment,
    so PROVIDER and the API key must be forwarded explicitly -- and nothing
    unrelated should leak to the tool servers."""
    with _env(PROVIDER="gemini", GEMINI_API_KEY="test-key-not-real", UNRELATED_SECRET="nope"):
        env = _server_env()
    assert env["PROVIDER"] == "gemini"
    assert env["GEMINI_API_KEY"] == "test-key-not-real"
    assert "UNRELATED_SECRET" not in env


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} gemini provider tests passed")
