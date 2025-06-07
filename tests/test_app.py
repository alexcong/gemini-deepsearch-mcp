"""Tests for the FastMCP server in src/app.py."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from google.api_core import exceptions as google_exceptions
from langchain_core.messages import AIMessage, HumanMessage

from gemini_deepsearch_mcp.app import app, deep_search

pytestmark = pytest.mark.anyio


@pytest.fixture(scope="session")
def anyio_backend():
    """Use only asyncio backend for tests."""
    return "asyncio"


@pytest.fixture
def mock_graph_result():
    """Mock graph result with answer and sources."""
    return {
        "messages": [
            HumanMessage(content="What is climate change?"),
            AIMessage(
                content="Climate change refers to long-term shifts in global temperatures and weather patterns."
            ),
        ],
        "sources_gathered": [
            {"url": "https://example.com/climate", "title": "Climate Change Overview"},
            {"url": "https://example.com/science", "title": "Climate Science"},
        ],
    }


class TestDeepSearchTool:
    """Test cases for the deep_search tool function."""

    async def test_deep_search_low_effort(self, mock_graph_result):
        """Test deep_search with low effort level."""
        with (
            patch(
                "gemini_deepsearch_mcp.app.graph.invoke", return_value=mock_graph_result
            ),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
        ):
            mock_to_thread.return_value = mock_graph_result

            result = await deep_search.fn("What is climate change?", "low")

            # Verify result structure
            assert "answer" in result
            assert "sources" in result
            assert (
                result["answer"]
                == "Climate change refers to long-term shifts in global temperatures and weather patterns."
            )
            assert len(result["sources"]) == 2

            # Verify asyncio.to_thread was called
            mock_to_thread.assert_called_once()

            # Get the actual arguments passed to graph.invoke via asyncio.to_thread
            args, kwargs = mock_to_thread.call_args
            invoke_func, input_state, config = args

            # Verify low effort configuration
            assert input_state["initial_search_query_count"] == 1
            assert input_state["max_research_loops"] == 1
            assert input_state["reasoning_model"] == "gemini-2.5-flash-preview-05-20"
            assert len(input_state["messages"]) == 1
            assert input_state["messages"][0].content == "What is climate change?"

    async def test_deep_search_medium_effort(self, mock_graph_result):
        """Test deep_search with medium effort level."""
        with (
            patch(
                "gemini_deepsearch_mcp.app.graph.invoke", return_value=mock_graph_result
            ),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
        ):
            mock_to_thread.return_value = mock_graph_result

            result = await deep_search.fn("What is artificial intelligence?", "medium")

            # Verify result structure
            assert "answer" in result
            assert "sources" in result

            # Get the actual arguments passed to graph.invoke via asyncio.to_thread
            args, kwargs = mock_to_thread.call_args
            invoke_func, input_state, config = args

            # Verify medium effort configuration
            assert input_state["initial_search_query_count"] == 3
            assert input_state["max_research_loops"] == 2
            assert input_state["reasoning_model"] == "gemini-2.5-flash-preview-05-20"

    async def test_deep_search_high_effort(self, mock_graph_result):
        """Test deep_search with high effort level."""
        with (
            patch(
                "gemini_deepsearch_mcp.app.graph.invoke", return_value=mock_graph_result
            ),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
        ):
            mock_to_thread.return_value = mock_graph_result

            result = await deep_search.fn("Explain quantum computing", "high")

            # Verify result structure
            assert "answer" in result
            assert "sources" in result

            # Get the actual arguments passed to graph.invoke via asyncio.to_thread
            args, kwargs = mock_to_thread.call_args
            invoke_func, input_state, config = args

            # Verify high effort configuration
            assert input_state["initial_search_query_count"] == 5
            assert input_state["max_research_loops"] == 3
            assert input_state["reasoning_model"] == "gemini-2.5-pro-preview-05-06"

    async def test_deep_search_default_effort(self, mock_graph_result):
        """Test deep_search with default effort level (should be low)."""
        with (
            patch(
                "gemini_deepsearch_mcp.app.graph.invoke", return_value=mock_graph_result
            ),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
        ):
            mock_to_thread.return_value = mock_graph_result

            await deep_search.fn("What is machine learning?")

            # Get the actual arguments passed to graph.invoke via asyncio.to_thread
            args, kwargs = mock_to_thread.call_args
            invoke_func, input_state, config = args

            # Verify default (low) effort configuration
            assert input_state["initial_search_query_count"] == 1
            assert input_state["max_research_loops"] == 1
            assert input_state["reasoning_model"] == "gemini-2.5-flash-preview-05-20"

    async def test_deep_search_empty_messages(self):
        """Test deep_search when graph returns empty messages."""
        mock_result = {"messages": [], "sources_gathered": []}

        with (
            patch("gemini_deepsearch_mcp.app.graph.invoke", return_value=mock_result),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
        ):
            mock_to_thread.return_value = mock_result

            result = await deep_search.fn("Test query", "low")

            assert result["answer"] == "No answer generated."
            assert result["sources"] == []

    async def test_deep_search_config_models(self, mock_graph_result):
        """Test that deep_search passes correct model configuration."""
        with (
            patch(
                "gemini_deepsearch_mcp.app.graph.invoke", return_value=mock_graph_result
            ),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
        ):
            mock_to_thread.return_value = mock_graph_result

            await deep_search.fn("Test query", "low")

            # Get the config passed to graph.invoke
            args, kwargs = mock_to_thread.call_args
            invoke_func, input_state, config = args

            # Verify model configuration
            expected_config = {
                "configurable": {
                    "query_generator_model": "gemini-2.5-flash-preview-05-20",
                    "reflection_model": "gemini-2.5-flash-preview-05-20",
                    "answer_model": "gemini-2.5-pro-preview-05-06",
                }
            }
            assert config == expected_config

    async def test_deep_search_input_state_structure(self, mock_graph_result):
        """Test that deep_search creates correct input state structure."""
        with (
            patch(
                "gemini_deepsearch_mcp.app.graph.invoke", return_value=mock_graph_result
            ),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
        ):
            mock_to_thread.return_value = mock_graph_result

            query = "What is renewable energy?"
            await deep_search.fn(query, "medium")

            # Get the input state passed to graph.invoke
            args, kwargs = mock_to_thread.call_args
            invoke_func, input_state, config = args

            # Verify input state structure
            assert "messages" in input_state
            assert "search_query" in input_state
            assert "web_research_result" in input_state
            assert "sources_gathered" in input_state
            assert "initial_search_query_count" in input_state
            assert "max_research_loops" in input_state
            assert "reasoning_model" in input_state

            # Verify initial values
            assert len(input_state["messages"]) == 1
            assert isinstance(input_state["messages"][0], HumanMessage)
            assert input_state["messages"][0].content == query
            assert input_state["search_query"] == []
            assert input_state["web_research_result"] == []
            assert input_state["sources_gathered"] == []


class TestFastAPIApp:
    """Test cases for the FastAPI application."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    def test_app_creation(self, client):
        """Test that the FastAPI app is created correctly."""
        # Test that the app responds (even if MCP endpoints aren't directly accessible)
        # The app should at least be instantiated without errors
        assert client.app is not None

    def test_mcp_mount(self, client):
        """Test that MCP server is mounted correctly."""
        # The MCP server should be mounted at /mcp-server
        # We can't easily test the MCP endpoints without full integration,
        # but we can verify the mount exists
        # Look for the mounted MCP server path
        mcp_mounted = any("/mcp-server" in str(route) for route in client.app.routes)
        assert mcp_mounted


class TestErrorHandling:
    """Test error handling scenarios."""

    async def test_deep_search_graph_exception(self):
        """Test deep_search when graph.invoke raises a generic Exception."""
        error_message = "Graph execution failed"
        with patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = Exception(error_message)

            result = await deep_search.fn("Test query for generic exception", "low")
            assert result == {
                "answer": "An unexpected error occurred during the search process. Please check logs for details.",
                "sources": [],
            }

    async def test_deep_search_google_api_permission_denied(self):
        """Test deep_search when graph.invoke raises PermissionDenied."""
        error_message = "Mocked Permission Denied"
        with patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = google_exceptions.PermissionDenied(
                error_message
            )

            result = await deep_search.fn("Test query for permission denied", "low")
            assert result == {
                "answer": f"A Google API error occurred: {error_message}. Please check logs for details.",
                "sources": [],
            }

    async def test_deep_search_google_api_service_unavailable(self):
        """Test deep_search when graph.invoke raises ServiceUnavailable."""
        error_message = "Mocked Service Unavailable"
        with patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.side_effect = google_exceptions.ServiceUnavailable(
                error_message
            )

            result = await deep_search.fn(
                "Test query for service unavailable", "low"
            )
            assert result == {
                "answer": f"A Google API error occurred: {error_message}. Please check logs for details.",
                "sources": [],
            }

    async def test_deep_search_invalid_effort_level(self, mock_graph_result):
        """Test deep_search with invalid effort level (should default to high)."""
        with (
            patch(
                "gemini_deepsearch_mcp.app.graph.invoke", return_value=mock_graph_result
            ),
            patch("gemini_deepsearch_mcp.app.asyncio.to_thread") as mock_to_thread,
        ):
            mock_to_thread.return_value = mock_graph_result

            # Pass an invalid effort level - should default to high effort
            await deep_search.fn("Test query", "invalid")

            # Get the actual arguments passed to graph.invoke via asyncio.to_thread
            args, kwargs = mock_to_thread.call_args
            invoke_func, input_state, config = args

            # Should default to high effort configuration
            assert input_state["initial_search_query_count"] == 5
            assert input_state["max_research_loops"] == 3
            assert input_state["reasoning_model"] == "gemini-2.5-pro-preview-05-06"
