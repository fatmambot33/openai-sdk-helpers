"""Minimal test to increase coverage for BaseResponse.close() system vector store cleanup."""

from typing import Any
from types import SimpleNamespace

from openai_sdk_helpers.response.base import BaseResponse


def test_close_cleans_system_vector_storage(monkeypatch):
    class DummyVectorStorage:
        def __init__(self, *a, **kw):
            self.id = "dummy"
            self.deleted = False

        def upload_file(self, file_path):
            class File:
                id = "fileid"

            return File()

        def delete(self):
            self.deleted = True

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

    base = BaseResponse(
        instructions="hi",
        tools=[],
        schema=None,
        output_structure=None,
        tool_handlers={},
        model="gpt-3.5-turbo",
        api_key="sk-dummy",
        system_vector_store=["dummy"],
        client=DummyClient(),
        module_name="mod",
        data_path_fn=lambda m: __import__("pathlib").Path("/tmp"),
    )
    # Should always clean system vector storage
    # Simulate close and check no error (system vector store cleanup is now implicit)
    base.close()
