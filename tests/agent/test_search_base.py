"""Tests for generic search agent base classes."""

from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from openai_sdk_helpers.agent.search.base import (
    SearchPlanner,
    SearchToolAgent,
    SearchWriter,
)
from openai_sdk_helpers.agent.config import AgentConfig


# Mock Pydantic models for testing
class MockPlanStructure(BaseModel):
    """Mock search plan structure."""

    searches: List[MockItemStructure] = []


class MockItemStructure(BaseModel):
    """Mock search item structure."""

    query: str = "test query"


class MockResultStructure(BaseModel):
    """Mock search result structure."""

    text: str = "test result"


class MockReportStructure(BaseModel):
    """Mock search report structure."""

    report: str = "test report"


# Concrete implementations for testing
class TestSearchPlanner(SearchPlanner[MockPlanStructure]):
    """Concrete planner implementation for testing."""

    def _configure_agent(self) -> AgentConfig:
        return AgentConfig(
            name="test_planner",
            description="Test planner",
            output_type=MockPlanStructure,
        )


class TestSearchToolAgent(
    SearchToolAgent[MockItemStructure, MockResultStructure, MockPlanStructure]
):
    """Concrete tool agent implementation for testing."""

    def _configure_agent(self) -> AgentConfig:
        return AgentConfig(
            name="test_tool",
            description="Test tool",
            input_type=MockPlanStructure,
        )

    async def run_search(self, item: MockItemStructure) -> MockResultStructure:
        await asyncio.sleep(0.01)  # Simulate async work
        return MockResultStructure(text=f"result for {item.query}")


class TestSearchWriter(SearchWriter[MockReportStructure]):
    """Concrete writer implementation for testing."""

    def _configure_agent(self) -> AgentConfig:
        return AgentConfig(
            name="test_writer",
            description="Test writer",
            output_type=MockReportStructure,
        )


class TestSearchPlannerClass:
    """Test SearchPlanner generic base class."""

    @pytest.mark.asyncio
    async def test_planner_initialization(self) -> None:
        """Test planner agent initialization with default model."""
        planner = TestSearchPlanner(default_model="gpt-4o-mini")
        assert planner.agent_name == "test_planner"
        assert planner._output_type == MockPlanStructure

    @pytest.mark.asyncio
    async def test_planner_run_agent(self) -> None:
        """Test planner run_agent calls run_async."""
        planner = TestSearchPlanner(default_model="gpt-4o-mini")
        mock_plan = MockPlanStructure(searches=[MockItemStructure(query="q1")])

        with patch.object(
            planner, "run_async", new_callable=AsyncMock, return_value=mock_plan
        ) as mock_run_async:
            result = await planner.run_agent("test query")
            mock_run_async.assert_called_once()
            assert result == mock_plan

    @pytest.mark.asyncio
    async def test_planner_configure_agent_called(self) -> None:
        """Test that _configure_agent is called during init."""
        with patch.object(
            TestSearchPlanner,
            "_configure_agent",
            return_value=AgentConfig(
                name="test_planner",
                description="Test planner",
                output_type=MockPlanStructure,
            ),
        ) as mock_config:
            planner = TestSearchPlanner(default_model="gpt-4o-mini")
            mock_config.assert_called_once()


class TestSearchToolAgentClass:
    """Test SearchToolAgent generic base class."""

    @pytest.mark.asyncio
    async def test_tool_initialization(self) -> None:
        """Test tool agent initialization."""
        tool = TestSearchToolAgent(
            default_model="gpt-4o-mini", max_concurrent_searches=5
        )
        assert tool._max_concurrent_searches == 5
        assert tool.agent_name == "test_tool"

    @pytest.mark.asyncio
    async def test_tool_run_agent_executes_searches(self) -> None:
        """Test tool agent executes all searches with concurrency control."""
        tool = TestSearchToolAgent(
            default_model="gpt-4o-mini", max_concurrent_searches=2
        )

        items = [MockItemStructure(query=f"query_{i}") for i in range(3)]
        plan = MockPlanStructure(searches=items)

        results = await tool.run_agent(plan)

        assert len(results) == 3
        assert all(isinstance(r, MockResultStructure) for r in results)
        assert results[0].text == "result for query_0"
        assert results[1].text == "result for query_1"
        assert results[2].text == "result for query_2"

    @pytest.mark.asyncio
    async def test_tool_respects_concurrency_limit(self) -> None:
        """Test that tool respects max concurrent searches."""
        tool = TestSearchToolAgent(
            default_model="gpt-4o-mini", max_concurrent_searches=1
        )

        concurrent_count = 0
        max_concurrent = 0

        async def counting_search(item: MockItemStructure) -> MockResultStructure:
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return MockResultStructure(text=f"result {item.query}")

        # Monkey patch the run_search method
        tool.run_search = counting_search  # type: ignore

        items = [MockItemStructure(query=f"q{i}") for i in range(3)]
        plan = MockPlanStructure(searches=items)
        await tool.run_agent(plan)

        assert max_concurrent <= 1

    @pytest.mark.asyncio
    async def test_tool_handles_empty_searches(self) -> None:
        """Test tool agent handles empty search list."""
        tool = TestSearchToolAgent(default_model="gpt-4o-mini")
        plan = MockPlanStructure(searches=[])

        results = await tool.run_agent(plan)

        assert results == []

    @pytest.mark.asyncio
    async def test_tool_filters_none_results(self) -> None:
        """Test tool agent filters out None results."""
        tool = TestSearchToolAgent(default_model="gpt-4o-mini")

        async def search_with_none(
            item: MockItemStructure,
        ) -> MockResultStructure | None:
            if "skip" in item.query:
                return None
            return MockResultStructure(text=f"result {item.query}")

        tool.run_search = search_with_none  # type: ignore

        items = [
            MockItemStructure(query="q1"),
            MockItemStructure(query="skip_this"),
            MockItemStructure(query="q3"),
        ]
        plan = MockPlanStructure(searches=items)

        results = await tool.run_agent(plan)

        assert len(results) == 2
        assert results[0].text == "result q1"
        assert results[1].text == "result q3"


class TestSearchWriterClass:
    """Test SearchWriter generic base class."""

    @pytest.mark.asyncio
    async def test_writer_initialization(self) -> None:
        """Test writer agent initialization."""
        writer = TestSearchWriter(default_model="gpt-4o-mini")
        assert writer.agent_name == "test_writer"
        assert writer._output_type == MockReportStructure

    @pytest.mark.asyncio
    async def test_writer_run_agent(self) -> None:
        """Test writer run_agent passes correct context."""
        writer = TestSearchWriter(default_model="gpt-4o-mini")
        mock_report = MockReportStructure(report="final report")
        results = [
            MockResultStructure(text="r1"),
            MockResultStructure(text="r2"),
        ]

        with patch.object(
            writer, "run_async", new_callable=AsyncMock, return_value=mock_report
        ) as mock_run:
            result = await writer.run_agent("test query", results)

            # Verify run_async was called with correct arguments
            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs["input"] == "test query"
            assert "original_query" in call_kwargs["context"]
            assert "search_results" in call_kwargs["context"]
            assert call_kwargs["output_type"] == MockReportStructure
            assert result == mock_report

    @pytest.mark.asyncio
    async def test_writer_context_contains_search_results(self) -> None:
        """Test that writer passes search results in context."""
        writer = TestSearchWriter(default_model="gpt-4o-mini")
        results = [
            MockResultStructure(text="result 1"),
            MockResultStructure(text="result 2"),
        ]

        with patch.object(writer, "run_async", new_callable=AsyncMock) as mock_run:
            await writer.run_agent("query", results)

            context = mock_run.call_args[1]["context"]
            assert context["search_results"] == results
            assert context["original_query"] == "query"


class TestSearchAgentInheritance:
    """Test generic type inheritance patterns."""

    def test_planner_type_parameters(self) -> None:
        """Test that planner type parameters work correctly."""
        planner = TestSearchPlanner(default_model="gpt-4o-mini")
        # Type is correctly bound to MockPlanStructure
        assert planner._output_type == MockPlanStructure

    def test_tool_type_parameters(self) -> None:
        """Test that tool agent type parameters work correctly."""
        tool = TestSearchToolAgent(default_model="gpt-4o-mini")
        # Tool is properly typed with all three type variables
        assert tool.agent_name == "test_tool"
        assert tool._max_concurrent_searches == 10

    def test_writer_type_parameters(self) -> None:
        """Test that writer type parameters work correctly."""
        writer = TestSearchWriter(default_model="gpt-4o-mini")
        assert writer._output_type == MockReportStructure


class TestSearchAgentConcurrency:
    """Test concurrency behavior in search agents."""

    @pytest.mark.asyncio
    async def test_concurrent_search_execution(self) -> None:
        """Test searches execute concurrently with proper limits."""
        tool = TestSearchToolAgent(
            default_model="gpt-4o-mini", max_concurrent_searches=3
        )

        start_times = {}
        end_times = {}

        async def timed_search(item: MockItemStructure) -> MockResultStructure:
            start_times[item.query] = asyncio.get_event_loop().time()
            await asyncio.sleep(0.05)
            end_times[item.query] = asyncio.get_event_loop().time()
            return MockResultStructure(text=f"result {item.query}")

        tool.run_search = timed_search  # type: ignore

        items = [MockItemStructure(query=f"q{i}") for i in range(6)]
        plan = MockPlanStructure(searches=items)

        await tool.run_agent(plan)

        # Verify timing shows concurrent execution
        all_starts = sorted(start_times.values())
        all_ends = sorted(end_times.values())

        # With 6 items and concurrency of 3, should take at least 2 batches
        total_time = all_ends[-1] - all_starts[0]
        # Minimum time if perfectly concurrent: ~100ms (2 batches of 50ms each)
        assert total_time < 0.15  # Some overhead for concurrency management


class TestSearchAgentErrorHandling:
    """Test error handling in search agents."""

    @pytest.mark.asyncio
    async def test_writer_handles_list_result_types(self) -> None:
        """Test writer gracefully handles list of result types."""
        writer = TestSearchWriter(default_model="gpt-4o-mini")

        # List of results
        results = [
            MockResultStructure(text="result 1"),
            MockResultStructure(text="result 2"),
        ]

        with patch.object(writer, "run_async", new_callable=AsyncMock) as mock_run:
            await writer.run_agent("query", results)

            context = mock_run.call_args[1]["context"]
            assert len(context["search_results"]) == 2
