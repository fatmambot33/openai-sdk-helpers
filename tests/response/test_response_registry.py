"""Tests for ResponseRegistry and enhanced response configuration."""

from __future__ import annotations

import pytest

from openai_sdk_helpers.response import (
    ResponseConfiguration,
    ResponseRegistry,
    get_default_registry,
)


def test_response_registry_basic_operations():
    """Test basic registry operations: register, get, list."""
    registry = ResponseRegistry()

    # Create a simple configuration
    config = ResponseConfiguration(
        name="test_config",
        instructions="Test instructions",
        tools=None,
        input_structure=None,
        output_structure=None,
    )

    # Register configuration
    registry.register(config)

    # Retrieve configuration
    retrieved = registry.get("test_config")
    assert retrieved.name == "test_config"
    assert retrieved.instructions == "Test instructions"

    # List configurations
    names = registry.list_names()
    assert "test_config" in names
    assert len(names) == 1


def test_response_registry_duplicate_name_raises():
    """Test that registering duplicate names raises ValueError."""
    registry = ResponseRegistry()

    config1 = ResponseConfiguration(
        name="duplicate",
        instructions="First",
        tools=None,
        input_structure=None,
        output_structure=None,
    )

    config2 = ResponseConfiguration(
        name="duplicate",
        instructions="Second",
        tools=None,
        input_structure=None,
        output_structure=None,
    )

    registry.register(config1)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(config2)


def test_response_registry_get_nonexistent_raises():
    """Test that getting a nonexistent configuration raises KeyError."""
    registry = ResponseRegistry()

    with pytest.raises(KeyError, match="No configuration named"):
        registry.get("nonexistent")


def test_response_registry_clear():
    """Test that clear removes all configurations."""
    registry = ResponseRegistry()

    config = ResponseConfiguration(
        name="test",
        instructions="Test",
        tools=None,
        input_structure=None,
        output_structure=None,
    )

    registry.register(config)
    assert len(registry.list_names()) == 1

    registry.clear()
    assert len(registry.list_names()) == 0


def test_response_registry_multiple_configs():
    """Test registry with multiple configurations."""
    registry = ResponseRegistry()

    configs = [
        ResponseConfiguration(
            name=f"config_{i}",
            instructions=f"Instructions {i}",
            tools=None,
            input_structure=None,
            output_structure=None,
        )
        for i in range(5)
    ]

    for config in configs:
        registry.register(config)

    names = registry.list_names()
    assert len(names) == 5
    assert names == sorted([f"config_{i}" for i in range(5)])


def test_get_default_registry():
    """Test that get_default_registry returns a singleton."""
    registry1 = get_default_registry()
    registry2 = get_default_registry()

    assert registry1 is registry2  # Same instance

    # Add a config to verify it persists
    config = ResponseConfiguration(
        name="singleton_test",
        instructions="Test",
        tools=None,
        input_structure=None,
        output_structure=None,
    )

    registry1.register(config)
    assert "singleton_test" in registry2.list_names()

    # Clean up
    registry1.clear()


def test_response_registry_isolated_instances():
    """Test that separate registry instances are independent."""
    registry1 = ResponseRegistry()
    registry2 = ResponseRegistry()

    config1 = ResponseConfiguration(
        name="config1",
        instructions="First",
        tools=None,
        input_structure=None,
        output_structure=None,
    )

    config2 = ResponseConfiguration(
        name="config2",
        instructions="Second",
        tools=None,
        input_structure=None,
        output_structure=None,
    )

    registry1.register(config1)
    registry2.register(config2)

    # Each registry should only have its own config
    assert registry1.list_names() == ["config1"]
    assert registry2.list_names() == ["config2"]
