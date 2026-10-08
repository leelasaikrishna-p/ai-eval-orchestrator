"""MCP server exposing `groundedness_check` as a callable tool over stdio."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from tools.groundedness_check import groundedness_check  # noqa: E402

mcp = MCPServer("groundedness-check")


@mcp.tool()
def groundedness_check_tool(source_context: str, candidate_answer: str) -> dict:
    """Check whether every factual claim in `candidate_answer` is actually
    supported by `source_context`, flagging anything the answer asserts
    that goes beyond what the context backs up.

    Use this to catch hallucination in RAG-style answers -- a claim can
    sound plausible and still not be grounded in the retrieved source.
    """
    # Parameter names match the other tools and the orchestrator's prompt
    # labels: models fill tool arguments from those labels, and live runs
    # failed when this tool alone was named (context, answer).
    return groundedness_check(source_context, candidate_answer)


if __name__ == "__main__":
    mcp.run(transport="stdio")
