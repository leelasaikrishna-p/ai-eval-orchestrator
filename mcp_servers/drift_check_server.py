"""MCP server exposing `drift_check` as a callable tool over stdio.

Unlike the other three tools, this one takes explicit named metrics rather
than a raw dict, so the generated tool schema is self-documenting for
whatever MCP client calls it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from tools.drift_check import drift_check  # noqa: E402

mcp = MCPServer("drift-check")


@mcp.tool()
def drift_check_tool(
    avg_golden_similarity: float,
    avg_groundedness: float,
    avg_judge_score: float,
) -> dict:
    """Compare this run's aggregate evaluation scores against the stored
    historical baseline, flagging any metric that regressed more than 10%.

    Pass the AVERAGE golden_eval score, AVERAGE groundedness pass rate, and
    AVERAGE llm_judge score across a batch of evaluated answers -- this
    gates the health of the whole run/deployment, not a single answer.
    No model call involved; this is pure arithmetic against data/baseline_scores.json.
    """
    current = {
        "avg_golden_similarity": avg_golden_similarity,
        "avg_groundedness": avg_groundedness,
        "avg_judge_score": avg_judge_score,
    }
    return drift_check(current)


if __name__ == "__main__":
    mcp.run(transport="stdio")
