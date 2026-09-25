"""
End-to-end MCP protocol tests: spawns each server as a real subprocess over
stdio (exactly how a real MCP client -- Claude Desktop, or our own future
orchestrator -- would talk to it) and calls its tool for real, rather than
just importing the module and calling the Python function directly.

drift_check_server needs no LLM, so it's tested against real Ollama-free
output. The other three are exercised with PROVIDER=mock so this test
suite runs fast and offline; a separate manual run against real Ollama
already validated actual judgment quality (see README).
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402

ROOT = Path(__file__).parent.parent
PYTHON = sys.executable


async def call_tool(script: str, tool_name: str, arguments: dict, env_extra: dict | None = None) -> dict:
    env = {**os.environ, **(env_extra or {})}
    params = StdioServerParameters(command=PYTHON, args=[str(ROOT / "mcp_servers" / script)], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert any(t.name == tool_name for t in tools.tools), (
                f"{tool_name} not found in {[t.name for t in tools.tools]}"
            )
            result = await session.call_tool(tool_name, arguments)
            assert not result.is_error, f"Tool call failed: {result.content}"
            # our tools return plain dicts, so no output schema was registered --
            # the result comes back as a JSON text block rather than structured_content
            return json.loads(result.content[0].text)


async def _test_drift_check_server():
    result = await call_tool(
        "drift_check_server.py",
        "drift_check_tool",
        {"avg_golden_similarity": 0.85, "avg_groundedness": 0.70, "avg_judge_score": 8.0},
    )
    assert result["drifted"] is True
    print("  ok  drift_check_server responds correctly over real MCP stdio")


async def _test_golden_eval_server_with_mock():
    result = await call_tool(
        "golden_eval_server.py",
        "golden_eval_tool",
        {"question": "q?", "golden_answer": "a", "candidate_answer": "a"},
        env_extra={"PROVIDER": "mock"},
    )
    assert "score" in result
    print("  ok  golden_eval_server responds correctly over real MCP stdio (mock provider)")


async def _test_groundedness_check_server_with_mock():
    result = await call_tool(
        "groundedness_check_server.py",
        "groundedness_check_tool",
        {"context": "c", "answer": "a"},
        env_extra={
            "PROVIDER": "mock",
            "MOCK_RESPONSE": '{"grounded": true, "unsupported_claims": [], "reasoning": "ok"}',
        },
    )
    assert result["grounded"] is True
    print("  ok  groundedness_check_server responds correctly over real MCP stdio (mock provider)")


async def _test_llm_judge_server_with_mock():
    result = await call_tool(
        "llm_judge_server.py",
        "llm_judge_tool",
        {"question": "q?", "answer": "a"},
        env_extra={
            "PROVIDER": "mock",
            "MOCK_RESPONSE": '{"score": 7.5, "breakdown": {"correctness": 8, "clarity": 7, "tone": 7}, "reasoning": "ok"}',
        },
    )
    assert result["score"] == 7.5
    print("  ok  llm_judge_server responds correctly over real MCP stdio (mock provider)")


async def main():
    await _test_drift_check_server()
    await _test_golden_eval_server_with_mock()
    await _test_groundedness_check_server_with_mock()
    await _test_llm_judge_server_with_mock()
    print("\n4/4 MCP protocol tests passed")


if __name__ == "__main__":
    asyncio.run(main())
