import asyncio
import logging
from typing import Annotated, Literal

from fastapi import FastAPI
from fastmcp import FastMCP
from google.api_core import exceptions as google_exceptions
from langchain_core.messages import HumanMessage
from pydantic import Field
from starlette.routing import Mount

from .agent.configuration import Configuration
from .agent.graph import graph
from .agent.utils import EffortSettings, get_effort_settings

mcp = FastMCP("DeepSearch")


def _prepare_agent_input_state(query: str, effort_settings: EffortSettings) -> dict:
    """Prepares the input state for the agent graph."""
    return {
        "messages": [HumanMessage(content=query)],
        "search_query": [],
        "web_research_result": [],
        "sources_gathered": [],
        "initial_search_query_count": effort_settings["initial_search_query_count"],
        "max_research_loops": effort_settings["max_research_loops"],
        "reasoning_model": effort_settings["reasoning_model"],
    }


@mcp.tool()
async def deep_search(
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

    # Prepare the input state
    input_state = _prepare_agent_input_state(query, effort_settings)

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
    try:
        result = await asyncio.to_thread(graph.invoke, input_state, config)

        # Extract the final answer and sources from the result
        answer = (
            result["messages"][-1].content
            if result["messages"]
            else "No answer generated."
        )
        sources = result["sources_gathered"]
        return {"answer": answer, "sources": sources}
    except google_exceptions.GoogleAPIError as e:
        logging.error(f"Google API error during deep search: {e}", exc_info=True)
        return {
            "answer": f"A Google API error occurred: {e.message}. Please check logs for details.",
            "sources": [],
        }
    except Exception as e:
        logging.error(f"Unexpected error during deep search: {e}", exc_info=True)
        return {
            "answer": "An unexpected error occurred during the search process. Please check logs for details.",
            "sources": [],
        }


# Create the ASGI app
mcp_app = mcp.http_app(path="/mcp")

# Create a FastAPI app and mount the MCP server
app = FastAPI(lifespan=mcp_app.lifespan)
app.mount("/mcp-server", mcp_app)
