"""Main entry point for the stdio MCP server."""

import sys
from typing import Annotated, Literal

from fastmcp import FastMCP
from langchain_core.messages import HumanMessage
from pydantic import Field

from .agent.configuration import Configuration
from .agent.graph import graph
from .agent.utils import get_effort_settings

# Create MCP server
mcp = FastMCP("DeepSearch")


@mcp.tool()
def deep_search(
    query: Annotated[str, Field(description="Search query string")],
    effort: Annotated[
        Literal["low", "medium", "high"], Field(description="Search effort")
    ] = "low",
) -> dict:
    """Perform a deep search on a given query using an advanced web research agent.

    Args:
        query: The research question or topic to investigate.
        effort: The amount of effect for the research, low, medium or hight (default: low).

    Returns:
        A dictionary containing the answer to the query and a list of sources used.
    """
    # Get effort settings
    effort_settings = get_effort_settings(effort)

    # Prepare the input state with the user's query
    input_state = {
        "messages": [HumanMessage(content=query)],
        "search_query": [],
        "web_research_result": [],
        "sources_gathered": [],
        "initial_search_query_count": effort_settings["initial_search_query_count"],
        "max_research_loops": effort_settings["max_research_loops"],
        "reasoning_model": effort_settings["reasoning_model"],
    }

    agent_config = Configuration()
    # Configuration for the agent
    config = {
        "configurable": {
            "query_generator_model": agent_config.query_generator_model,
            "reflection_model": agent_config.reflection_model,
            "answer_model": agent_config.answer_model,
        }
    }

    # Run the agent graph to process the query in a separate thread to avoid blocking
    result = graph.invoke(input_state, config)

    # Extract the final answer and sources from the result
    answer = (
        result["messages"][-1].content if result["messages"] else "No answer generated."
    )
    sources = result["sources_gathered"]

    return {"answer": answer, "sources": sources}


def main():
    """Main entry point for the MCP server."""
    sys.stderr.write("Starting MCP stdio server...\n")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
