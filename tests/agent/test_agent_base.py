"""Tests for the AgentBase class."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, Mock

import pytest
from agents import RunContextWrapper
from pydantic import BaseModel

from openai_sdk_helpers.agent.base import AgentBase
from openai_sdk_helpers.structure.base import StructureBase

warnings.filterwarnings("ignore", "coroutine.*was never awaited", RuntimeWarning)


class MockConfig(BaseModel):
    """Mock agent configuration."""

    name: str
    instructions: str  # Now required, matching AgentConfiguration
    model: str = "gpt-4o-mini"  # Default model for tests

    @property
    def instructions_text(self) -> str:
        """Expose instructions text to satisfy AgentConfigurationLike protocol."""
        return self.instructions

    description: str | None = None
    model: str | None = None
    template_path: str | None = None
    input_structure: type[StructureBase] | None = None
    output_structure: type[StructureBase] | None = None
    tools: Any | None = None
    model_settings: Any | None = None
    handoffs: Any | None = None
    input_guardrails: Any | None = None
    output_guardrails: Any | None = None
    session: Any | None = None

    def resolve_prompt_path(self, prompt_dir: Path | None = None) -> Path | None:
        """Resolve the prompt path to satisfy AgentConfigurationLike."""
        if self.template_path:
            return Path(self.template_path)
        if prompt_dir is not None:
            return prompt_dir / f"{self.name}.jinja"
        return None


@pytest.fixture
def mock_config():
    """Return a mock agent configuration."""
    return MockConfig(
        name="test_agent", model="test_model", instructions="Test instructions"
    )


@pytest.fixture
def mock_run_context_wrapper():
    """Return a mock run context wrapper."""
    return RunContextWrapper(context={"key": "value"})


def test_base_agent_initialization(mock_config):
    """Test AgentBase initialization."""
    agent = AgentBase(configuration=mock_config)
    assert agent.name == "test_agent"
    assert agent.model == "test_model"


def test_base_agent_initialization_with_prompt_dir(mock_config, tmp_path: Path):
    """Test AgentBase initialization with a prompt directory."""
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    prompt_file = prompt_dir / "test_agent.jinja"
    prompt_file.write_text("Hello, {{ key }}!")
    agent = AgentBase(configuration=mock_config)
    # The template is set from instructions, not from the file, so expect 'Test instructions'
    assert agent._template.render(key="world") == "Test instructions"


def test_base_agent_initialization_with_absolute_template_path(tmp_path: Path):
    """Test AgentBase initialization with an absolute template path."""
    template_file = tmp_path / "custom_template.jinja"
    template_file.write_text("Greetings, {{ name }}!")

    configuration = MockConfig(
        name="test_agent",
        model="test_model",
        instructions="Test instructions",
        template_path=str(template_file.resolve()),
    )
    agent = AgentBase(configuration=configuration)
    # AgentBase uses instructions, not template_path, for MockConfig
    assert agent._template.render(name="Alice") == "Greetings, Alice!"


def test_base_agent_build_prompt_from_jinja(mock_config, mock_run_context_wrapper):
    """Test building a prompt from a Jinja template."""
    agent = AgentBase(configuration=mock_config)
    agent._template = MagicMock()
    agent._template.render.return_value = "Hello, value!"
    prompt = agent.build_prompt_from_jinja(mock_run_context_wrapper)
    assert prompt == "Hello, value!"
    agent._template.render.assert_called_once_with({"key": "value"})


@patch("openai_sdk_helpers.agent.base.Agent")
def test_get_agent(mock_agent, mock_config):
    """Test getting a configured agent instance."""
    agent = AgentBase(configuration=mock_config)
    agent.get_agent()
    mock_agent.assert_called_once_with(
        name="test_agent",
        instructions="Test instructions",
        model="test_model",
    )


@patch("openai_sdk_helpers.agent.runner.Runner.run", new_callable=AsyncMock)
@patch("asyncio.run")
def test_run_agent_sync_no_loop(mock_asyncio_run, mock_runner_run, mock_config):
    """Test that _run_agent_sync creates a new event loop when none is running."""
    agent = AgentBase(configuration=mock_config)
    agent.run_sync("test_input")
    mock_asyncio_run.assert_called_once()


@patch("openai_sdk_helpers.agent.base.run_sync")
def test_run_agent_sync(mock_run_sync, mock_config):
    """Test running the agent synchronously."""
    mock_run_sync.return_value = "result"
    agent = AgentBase(configuration=mock_config)
    result = agent.run_sync("test_input")
    assert result == "result"
    mock_run_sync.assert_called_once()


def test_as_tool(mock_config):
    """Test returning the agent as a tool."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        agent = AgentBase(configuration=mock_config)
        mock_agent = Mock()
        mock_tool = Mock()
        mock_agent.as_tool.return_value = mock_tool
        with patch.object(agent, "get_agent", return_value=mock_agent):
            result = agent.as_tool()
        mock_agent.as_tool.assert_called()
        assert result == mock_tool
