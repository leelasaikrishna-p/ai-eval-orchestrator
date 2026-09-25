"""
The orchestrator -- this is the actual agentic piece.

The three per-answer tools (golden_eval, groundedness_check, llm_judge) run
as MCP servers. This script connects to all three as a real MCP client,
hands their tool schemas to the model, and lets the MODEL decide which
tool(s) to call and in what order, based on results it's already seen --
not a hardcoded "always call all three in this order" pipeline. That
distinction is the entire point: an MCP-wrapped tool is not an agent; a
loop where the model decides what to call next, based on what it just
learned, is.

The model's job stops at gathering evidence. The final approve/escalate
verdict is computed by decision_policy.sign_off() -- a tested, deterministic
function -- not the model's own free-form opinion. Agentic investigation,
governed decision: the model decides HOW to investigate, tested code
decides the VERDICT.
"""
from __future__ import annotations

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

sys.path.insert(0, str(Path(__file__).parent))
from decision_policy import sign_off  # noqa: E402
from data_utils import load_golden_dataset, load_sample_candidate_runs  # noqa: E402
from llm_provider import OllamaProvider  # noqa: E402

ROOT = Path(__file__).parent
SERVERS = {
    "golden_eval_tool": "golden_eval_server.py",
    "groundedness_check_tool": "groundedness_check_server.py",
    "llm_judge_tool": "llm_judge_server.py",
}
MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = """You are evaluating whether a candidate answer to a customer's \
question is trustworthy enough to ship. You have three tools:

- golden_eval_tool: scores semantic similarity to a known-correct reference answer (0.0-1.0)
- groundedness_check_tool: checks whether the answer's claims are actually supported by the source context
- llm_judge_tool: rates general answer quality on a 0-10 rubric

Call whichever tools you need, in whatever order makes sense, to decide if \
this answer is safe to approve. You do not need to call every tool -- for \
example, if golden_eval_tool already returns a very low score (well under \
0.3), that alone is enough evidence and you can stop there without calling \
the others. But if the answer looks reasonable so far, call groundedness_check_tool \
and llm_judge_tool too before concluding -- a good semantic-similarity score alone \
doesn't rule out a hallucinated or low-quality answer.

Once you've gathered enough evidence, respond with a short plain-text summary \
of what you found (no more tool calls).
"""


async def _connect_all(stack: AsyncExitStack) -> dict[str, ClientSession]:
    """Spawns each MCP server and returns {tool_name: open ClientSession}."""
    sessions = {}
    for tool_name, script in SERVERS.items():
        params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_servers" / script)])
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        sessions[tool_name] = session
    return sessions


async def _mcp_tools_as_ollama_schema(sessions: dict[str, ClientSession]) -> list[dict]:
    """Fetches each server's real tool schema (not hand-copied) and converts
    it to Ollama's function-calling format."""
    schemas = []
    for tool_name, session in sessions.items():
        listed = await session.list_tools()
        tool = next(t for t in listed.tools if t.name == tool_name)
        schemas.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        })
    return schemas


async def evaluate(question: str, golden_answer: str, context: str, candidate_answer: str) -> dict:
    """Runs the full agentic loop for one candidate answer. Returns the
    trace of tool calls made, the model's summary, and the final verdict."""
    provider = OllamaProvider()
    trace: list[dict] = []
    collected: dict[str, dict] = {}

    async with AsyncExitStack() as stack:
        sessions = await _connect_all(stack)
        tools_schema = await _mcp_tools_as_ollama_schema(sessions)

        user_prompt = (
            f"QUESTION: {question}\n\nGOLDEN ANSWER: {golden_answer}\n\n"
            f"SOURCE CONTEXT: {context}\n\nCANDIDATE ANSWER TO EVALUATE: {candidate_answer}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        summary = ""
        for _ in range(MAX_TOOL_ROUNDS):
            message = provider.chat(messages, tools=tools_schema)
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                summary = message.get("content", "")
                break

            messages.append(message)
            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"]["arguments"]
                session = sessions[name]
                result = await session.call_tool(name, args)
                if result.is_error:
                    # A failed call must NOT count as evidence -- leave it out
                    # of `collected` entirely so sign_off() sees None (missing),
                    # not a malformed dict it might crash trying to read.
                    error_text = result.content[0].text if result.content else "unknown error"
                    trace.append({"tool": name, "arguments": args, "result": {"error": error_text}})
                    messages.append({"role": "tool", "content": json.dumps({"error": error_text})})
                    continue
                result_data = json.loads(result.content[0].text)
                collected[name] = result_data
                trace.append({"tool": name, "arguments": args, "result": result_data})
                messages.append({"role": "tool", "content": json.dumps(result_data)})
        else:
            summary = "(stopped -- exceeded max tool-call rounds)"

    return {"trace": trace, "model_summary": summary, "verdict": verdict_from_collected(collected)}


def verdict_from_collected(collected: dict[str, dict]) -> dict:
    """Pure function: turns whatever tool results the agentic loop actually
    collected into a final verdict. Separated from `evaluate()` so this
    decision logic is unit-testable without spinning up MCP servers or a
    live model -- only the tool-calling loop above needs those."""
    if "golden_eval_tool" not in collected:
        return {
            "verdict": "escalate",
            "reasons": ["golden_eval was never called -- no basis for a verdict"],
            "flags": [],
        }
    return sign_off(
        golden_result=collected.get("golden_eval_tool"),
        groundedness_result=collected.get("groundedness_check_tool"),
        judge_result=collected.get("llm_judge_tool"),
    )


async def _demo():
    golden = load_golden_dataset()
    for run in load_sample_candidate_runs():
        item = golden[run["id"]]
        print(f"\n{'=' * 70}\n[{run['id']}] {run['label']}")
        print(f"Q: {item['question']}")
        print(f"Candidate: {run['candidate_answer']}\n")

        result = await evaluate(item["question"], item["golden_answer"], item["context"], run["candidate_answer"])

        print(f"Tool calls made ({len(result['trace'])}):")
        for step in result["trace"]:
            print(f"  -> {step['tool']}: {step['result']}")
        print(f"\nModel summary: {result['model_summary']}")
        print(f"\nVERDICT: {result['verdict']['verdict'].upper()}")
        if result["verdict"]["reasons"]:
            print(f"  reasons: {result['verdict']['reasons']}")


if __name__ == "__main__":
    asyncio.run(_demo())
