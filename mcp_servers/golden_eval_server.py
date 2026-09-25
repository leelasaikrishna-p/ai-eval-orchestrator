"""
MCP server exposing `golden_eval` as a callable tool over stdio.

Run directly to smoke-test: `python3 mcp_servers/golden_eval_server.py`
(it will just sit waiting for stdio input -- that's correct, an MCP client
is what drives it). To actually use it, point an MCP client's config at
this script, e.g. Claude Desktop's mcpServers config, or the orchestrator
built in M3.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from tools.golden_eval import golden_eval  # noqa: E402

mcp = MCPServer("golden-eval")


@mcp.tool()
def golden_eval_tool(question: str, golden_answer: str, candidate_answer: str) -> dict:
    """Score how closely a candidate answer matches a known-correct golden
    answer to the same question, on a 0.0-1.0 semantic-similarity scale.

    Use this as a regression check against known-good behavior -- not an
    exact-match test, but a judgment of whether the candidate means the
    same thing as the reference answer.
    """
    return golden_eval(question, golden_answer, candidate_answer)


if __name__ == "__main__":
    mcp.run(transport="stdio")
