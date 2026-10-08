"""MCP server exposing `llm_judge` as a callable tool over stdio."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from tools.llm_judge import llm_judge  # noqa: E402

mcp = MCPServer("llm-judge")


@mcp.tool()
def llm_judge_tool(question: str, candidate_answer: str, golden_answer: str | None = None) -> dict:
    """Grade an answer's overall quality on a 0-10 rubric across
    correctness, clarity, and tone.

    Pass the known-correct answer as `golden_answer` whenever you have
    one: correctness is then graded against it instead of the model's own
    beliefs, which measurably catches more wrong answers. Even so, this
    check alone is NOT sufficient to catch factual errors -- pair it with
    groundedness_check (see this project's decision_policy.py for why).
    """
    return llm_judge(question, candidate_answer, reference_answer=golden_answer)


if __name__ == "__main__":
    mcp.run(transport="stdio")
