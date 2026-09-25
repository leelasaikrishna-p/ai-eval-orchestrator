"""MCP server exposing `groundedness_check` as a callable tool over stdio."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from tools.groundedness_check import groundedness_check  # noqa: E402

mcp = MCPServer("groundedness-check")


@mcp.tool()
def groundedness_check_tool(context: str, answer: str) -> dict:
    """Check whether every factual claim in `answer` is actually supported
    by `context`, flagging anything the answer asserts that goes beyond
    what the context backs up.

    Use this to catch hallucination in RAG-style answers -- a claim can
    sound plausible and still not be grounded in the retrieved source.
    """
    return groundedness_check(context, answer)


if __name__ == "__main__":
    mcp.run(transport="stdio")
