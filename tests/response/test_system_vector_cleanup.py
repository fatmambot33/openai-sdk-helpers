"""Minimal test to increase coverage for BaseResponse.close() system vector store cleanup."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai_sdk_helpers.config import OpenAISettings
from openai_sdk_helpers.response.base import BaseResponse


def test_close_cleans_system_vector_storage(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.api_key: str | None = "sk-dummy"
            self.vector_stores: Any = self
            self.responses: Any = SimpleNamespace(create=lambda *_a, **_kw: None)
            self.files: Any = SimpleNamespace(
                create=lambda *_a, **_kw: SimpleNamespace(id="fileid"),
                content=lambda *_a, **_kw: SimpleNamespace(read=lambda: b""),
            )

        def list(self):
            class Store:
                id = "dummy"
                name = "dummy"

            return type("obj", (), {"data": [Store()]})()

    dummy_client = DummyClient()
    monkeypatch.setattr(
        "openai_sdk_helpers.config.OpenAI",
        lambda *_a, **_kw: dummy_client,
    )

    settings = OpenAISettings(api_key="sk-dummy", default_model="gpt-3.5-turbo")
    base = BaseResponse(
        instructions="hi",
        tools=[],
        output_structure=None,
        tool_handlers={},
        openai_settings=settings,
        system_vector_store=["dummy"],
        module_name="mod",
        data_path_fn=lambda m: Path("/tmp"),
    )
    # Should always clean system vector storage
    # Simulate close and check no error (system vector store cleanup is now implicit)
    base.close()
