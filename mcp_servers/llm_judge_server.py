"""MCP server exposing `llm_judge` as a callable tool over stdio."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from tools.llm_judge import llm_judge  # noqa: E402

mcp = MCPServer("llm-judge")


@mcp.tool()
def llm_judge_tool(question: str, answer: str) -> dict:
    """Grade an answer's overall quality on a 0-10 rubric across
    correctness, clarity, and tone -- independent of any reference answer.

    Use this for general quality scoring. Note: this check alone is NOT
    sufficient to catch factual errors against a source -- pair it with
    groundedness_check for that (see this project's decision_policy.py
    for why judge scores can miss real problems on their own).
    """
    return llm_judge(question, answer)


if __name__ == "__main__":
    mcp.run(transport="stdio")
