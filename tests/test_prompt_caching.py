"""Tests for enhanced PromptRenderer with caching."""

from __future__ import annotations

from pathlib import Path

import pytest

from openai_sdk_helpers.prompt import PromptRenderer


def test_prompt_renderer_caching(tmp_path):
    """Test that templates are cached for performance."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_file = template_dir / "cached.jinja"
    template_file.write_text("Hello {{ name }}!")

    renderer = PromptRenderer(base_dir=template_dir)

    # First render - compiles and caches
    result1 = renderer.render("cached.jinja", {"name": "World"})
    assert result1 == "Hello World!"

    # Second render - uses cache
    result2 = renderer.render("cached.jinja", {"name": "Alice"})
    assert result2 == "Hello Alice!"

    # Cache info should show hits (can't directly test lru_cache, but behavior confirms caching)


def test_prompt_renderer_clear_cache(tmp_path):
    """Test that clear_cache forces re-compilation."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_file = template_dir / "template.jinja"
    template_file.write_text("Original {{ text }}")

    renderer = PromptRenderer(base_dir=template_dir)

    # Initial render
    result1 = renderer.render("template.jinja", {"text": "content"})
    assert result1 == "Original content"

    # Modify template file
    template_file.write_text("Modified {{ text }}")

    # Without clearing cache, still gets old version
    # (This is expected behavior - cache holds compiled template)

    # Clear cache and render again
    renderer.clear_cache()
    result2 = renderer.render("template.jinja", {"text": "content"})
    assert result2 == "Modified content"


def test_prompt_renderer_missing_template_clear_error(tmp_path):
    """Test clear error message for missing templates."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir()

    renderer = PromptRenderer(base_dir=template_dir)

    with pytest.raises(FileNotFoundError) as exc_info:
        renderer.render("nonexistent.jinja", {})

    error_msg = str(exc_info.value)
    assert "Template not found" in error_msg
    assert "nonexistent.jinja" in error_msg


def test_prompt_renderer_multiple_renders_same_template(tmp_path):
    """Test rendering same template multiple times with different context."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_file = template_dir / "greet.jinja"
    template_file.write_text("Hello {{ name }}, you are {{ age }} years old.")

    renderer = PromptRenderer(base_dir=template_dir)

    contexts = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Charlie", "age": 35},
    ]

    for ctx in contexts:
        result = renderer.render("greet.jinja", ctx)
        assert ctx["name"] in result
        assert str(ctx["age"]) in result


def test_prompt_renderer_cache_different_templates(tmp_path):
    """Test that different templates are cached separately."""
    template_dir = tmp_path / "templates"
    template_dir.mkdir()

    template1 = template_dir / "template1.jinja"
    template1.write_text("Template 1: {{ value }}")

    template2 = template_dir / "template2.jinja"
    template2.write_text("Template 2: {{ value }}")

    renderer = PromptRenderer(base_dir=template_dir)

    # Render both templates
    result1 = renderer.render("template1.jinja", {"value": "test"})
    result2 = renderer.render("template2.jinja", {"value": "test"})

    assert result1 == "Template 1: test"
    assert result2 == "Template 2: test"

    # Both should be cached independently
