"""Tests for the vector storage helper."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from openai_sdk_helpers.vector_storage.storage import VectorStorage


class DummyFile(SimpleNamespace):
    pass


class DummyVectorStore(SimpleNamespace):
    pass


class DummyClient:
    def __init__(self) -> None:
        self.files_created: list[DummyFile] = []
        self.stores: list[DummyVectorStore] = []
        self.file_contents: dict[str, bytes] = {}

    @property
    def files(self):
        return SimpleNamespace(create=self._create_file, content=self._file_content)

    @property
    def vector_stores(self):
        return SimpleNamespace(
            list=self._list_stores,
            create=self._create_store,
            delete=self._delete_store,
            search=self._search,
            files=SimpleNamespace(
                list=self._list_store_files,
                create=self._attach_file,
                delete=self._delete_file,
                poll=lambda *_args, **_kwargs: None,
            ),
        )

    def _list_stores(self):
        return SimpleNamespace(data=self.stores)

    def _create_store(self, name: str):
        store = DummyVectorStore(id=f"store-{len(self.stores)}", name=name, files=[])
        self.stores.append(store)
        return store

    def _delete_store(self, store_id: str):
        self.stores = [s for s in self.stores if s.id != store_id]

    def _list_store_files(self, vector_store_id: str):
        store = next(s for s in self.stores if s.id == vector_store_id)
        return SimpleNamespace(data=list(store.files))

    def _attach_file(self, vector_store_id: str, file_id: str, attributes: dict):
        store = next(s for s in self.stores if s.id == vector_store_id)
        store.files.append(DummyFile(id=file_id, attributes=attributes))

    def _delete_file(self, *, vector_store_id: str, file_id: str):
        store = next(s for s in self.stores if s.id == vector_store_id)
        store.files = [f for f in store.files if f.id != file_id]

    def _search(self, *, vector_store_id: str, query: str, max_num_results: int):
        return SimpleNamespace(data=[])

    def _create_file(self, file, purpose: str):
        file_path, _file_data = file
        new_file = DummyFile(id=f"file-{len(self.files_created)}", path=file_path)
        self.file_contents[new_file.id] = _file_data
        self.files_created.append(new_file)
        return new_file

    def _file_content(self, file_id: str):
        content = self.file_contents.get(file_id, b"")
        return SimpleNamespace(read=lambda: content)


@pytest.fixture()
def dummy_client():
    return DummyClient()


def test_vector_storage_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValueError):
        VectorStorage(store_name="missing", client=None)


def test_vector_storage_requires_model(monkeypatch, dummy_client):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValueError):
        VectorStorage(store_name="missing-model", client=dummy_client)


def test_upload_and_delete_file(tmp_path, dummy_client, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    storage = VectorStorage("test-store", client=dummy_client)

    text_file = tmp_path / "example.txt"
    text_file.write_text("hello")

    info = storage.upload_file(str(text_file))
    assert info.status == "success"
    assert info.name == "example.txt"
    assert storage.existing_files["example.txt"] == info.id

    deleted = storage.delete_file(info.id)
    assert deleted.status == "success"
    assert storage.existing_files == {}


def test_upload_files_skips_existing(tmp_path, dummy_client, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    storage = VectorStorage("batch-store", client=dummy_client)

    first = tmp_path / "a.txt"
    first.write_text("first")
    second = tmp_path / "b.txt"
    second.write_text("second")

    storage.upload_file(str(first))
    stats = storage.upload_files([str(first), str(second)])

    assert stats.total == 1
    assert stats.success == 1
    assert "b.txt" in storage.existing_files


def test_summarize_handles_empty_results(dummy_client, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    storage = VectorStorage("search-store", client=dummy_client)

    summary = storage.summarize("query")
    assert summary is None


def test_download_files(tmp_path, dummy_client, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    storage = VectorStorage("download-store", client=dummy_client)

    file_path = tmp_path / "download_me.txt"
    file_path.write_text("content to download")

    upload = storage.upload_file(str(file_path))
    output_dir = tmp_path / "downloads"

    stats = storage.download_files(str(output_dir))

    downloaded_path = output_dir / file_path.name
    assert stats.total == 1
    assert stats.success == 1
    assert downloaded_path.read_text() == "content to download"


def test_delete_files_and_store_cleanup(tmp_path, dummy_client, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    storage = VectorStorage("cleanup-store", client=dummy_client)

    file_path = tmp_path / "delete_me.txt"
    file_path.write_text("bye")
    upload = storage.upload_file(str(file_path))

    stats = storage.delete_files([upload.id])
    assert stats.success == 1

    storage.delete()
    assert dummy_client.stores == []


def test_summarize_raises_when_results_exist(dummy_client, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    storage = VectorStorage("summary-store", client=dummy_client)

    def _search_with_results(*, vector_store_id: str, query: str, max_num_results: int):
        return SimpleNamespace(data=[{"id": 1}])

    dummy_client._search = _search_with_results
    with pytest.raises(RuntimeError):
        storage.summarize("query")
