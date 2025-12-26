"""Tests for the BaseAgent class."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from agents import RunContextWrapper
from pydantic import BaseModel

from corai.agent.base import BaseAgent


class MockConfig(BaseModel):
    """Mock agent configuration."""

    name: str
    description: str | None = None
    model: str | None = None
    template_path: str | None = None
    input_type: Any | None = None
    output_type: Any | None = None
    tools: Any | None = None
    model_settings: Any | None = None


@pytest.fixture
def mock_config():
    """Return a mock agent configuration."""
    return MockConfig(name="test_agent", model="test_model")


@pytest.fixture
def mock_run_context_wrapper():
    """Return a mock run context wrapper."""
    return RunContextWrapper(context={"key": "value"})


def test_base_agent_initialization(mock_config):
    """Test BaseAgent initialization."""
    agent = BaseAgent(config=mock_config)
    assert agent.agent_name == "test_agent"
    assert agent.model == "test_model"


def test_base_agent_initialization_with_prompt_dir(mock_config, tmp_path: Path):
    """Test BaseAgent initialization with a prompt directory."""
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    prompt_file = prompt_dir / "test_agent.jinja"
    prompt_file.write_text("Hello, {{ key }}!")
    agent = BaseAgent(config=mock_config, prompt_dir=prompt_dir)
    assert agent._template.render(key="world") == "Hello, world!"


def test_base_agent_build_prompt_from_jinja(mock_config, mock_run_context_wrapper):
    """Test building a prompt from a Jinja template."""
    agent = BaseAgent(config=mock_config)
    agent._template = MagicMock()
    agent._template.render.return_value = "Hello, value!"
    prompt = agent.build_prompt_from_jinja(mock_run_context_wrapper)
    assert prompt == "Hello, value!"
    agent._template.render.assert_called_once_with({"key": "value"})


@patch("corai.agent.base.Agent")
def test_get_agent(mock_agent, mock_config):
    """Test getting a configured agent instance."""
    agent = BaseAgent(config=mock_config)
    agent.get_agent()
    mock_agent.assert_called_once_with(
        name="test_agent",
        instructions="",
        model="test_model",
    )


@patch("corai.agent.base.Runner")
@patch("asyncio.run")
def test_run_agent_sync_no_loop(mock_asyncio_run, mock_runner, mock_config):
    """Test that _run_agent_sync creates a new event loop when none is running."""
    agent = BaseAgent(config=mock_config)
    agent.run_sync("test_input")
    mock_asyncio_run.assert_called_once()


@patch("corai.agent.base._run_agent_sync")
def test_run_agent_sync(mock_run_agent_sync, mock_config):
    """Test running the agent synchronously."""
    agent = BaseAgent(config=mock_config)
    agent.run_sync("test_input")
    mock_run_agent_sync.assert_called_once()


@patch("corai.agent.base._run_agent_streamed")
def test_run_agent_streamed(mock_run_agent_streamed, mock_config):
    """Test running the agent with streaming."""
    agent = BaseAgent(config=mock_config)
    agent.run_streamed("test_input")
    mock_run_agent_streamed.assert_called_once()


@patch("corai.agent.base.FunctionTool")
@patch("corai.agent.base.Agent")
def test_as_tool(mock_agent_class, mock_function_tool, mock_config):
    """Test returning the agent as a tool."""
    mock_agent_instance = mock_agent_class.return_value
    mock_agent_instance.as_tool.return_value = mock_function_tool
    agent = BaseAgent(config=mock_config)
    agent.as_tool()
    mock_agent_instance.as_tool.assert_called_once_with(
        tool_name="test_agent", tool_description=""
    )
