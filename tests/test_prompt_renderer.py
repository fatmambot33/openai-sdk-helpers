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


def test_prompt_renderer_defaults_to_package_dir():
    renderer = PromptRenderer()
    assert renderer.base_dir.exists()


def test_prompt_renderer_ships_builtin_templates():
    renderer = PromptRenderer()
    template_path = renderer.base_dir / "summarizer.jinja"

    assert template_path.exists()
    content = template_path.read_text().strip()

    assert "summarize" in content.lower()
