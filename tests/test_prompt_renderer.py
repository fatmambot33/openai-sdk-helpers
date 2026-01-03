"""Tests for prompt rendering utilities."""

from __future__ import annotations

from pathlib import Path

from openai_sdk_helpers.prompt import PromptRenderer


def test_prompt_renderer_renders_template(tmp_path):
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_file = template_dir / "example.jinja"
    template_file.write_text("Hello {{ name }}!")

    renderer = PromptRenderer(base_dir=template_dir)
    rendered = renderer.render("example.jinja", {"name": "World"})

    assert rendered == "Hello World!"
    assert renderer.base_dir == template_dir
    assert Path(renderer.base_dir, "example.jinja").exists()


def test_prompt_renderer_renders_absolute_path(tmp_path):
    template_file = tmp_path / "absolute_example.jinja"
    template_file.write_text("Greetings {{ name }}!")

    renderer = PromptRenderer()
    rendered = renderer.render(str(template_file), {"name": "Alice"})

    assert rendered == "Greetings Alice!"


def test_prompt_renderer_renders_absolute_path_without_base_dir(tmp_path):
    template_file = tmp_path / "no_base_dir.jinja"
    template_file.write_text("Welcome {{ name }}!")

    # Create renderer without explicit base_dir
    renderer = PromptRenderer()
    # Use absolute path - should work regardless of base_dir
    rendered = renderer.render(str(template_file.resolve()), {"name": "Bob"})

    assert rendered == "Welcome Bob!"


def test_prompt_renderer_defaults_to_package_dir():
    renderer = PromptRenderer()
    assert renderer.base_dir.exists()


def test_prompt_renderer_ships_builtin_templates():
    renderer = PromptRenderer()
    template_path = renderer.base_dir / "summarizer.jinja"

    assert template_path.exists()
    content = template_path.read_text().strip()

    assert "summarize" in content.lower()
