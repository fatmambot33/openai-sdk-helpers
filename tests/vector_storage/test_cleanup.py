"""Tests for vector storage cleanup helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from corai.vector_storage.cleanup import _delete_all_files, _delete_all_vector_stores


class _FakeFile(SimpleNamespace):
    pass


class _FakeVectorStore(SimpleNamespace):
    pass


class _FakeClient:
    def __init__(self, stores: list[_FakeVectorStore], files: list[_FakeFile]):
        self._stores = stores
        self._files = files

    @property
    def vector_stores(self):
        return SimpleNamespace(
            list=self._list_stores,
            delete=self._delete_store,
            files=SimpleNamespace(list=self._list_files, delete=self._delete_file),
        )

    @property
    def files(self):
        return SimpleNamespace(list=self._list_all_files, delete=self._delete_file)

    def _list_stores(self):
        return SimpleNamespace(data=self._stores)

    def _list_files(self, vector_store_id: str):
        store = next(s for s in self._stores if s.id == vector_store_id)
        return SimpleNamespace(data=list(store.files))

    def _delete_store(self, store_id: str):
        self._stores = [s for s in self._stores if s.id != store_id]

    def _delete_file(self, *, file_id: str, vector_store_id: str | None = None):
        if vector_store_id:
            for store in self._stores:
                store.files = [f for f in store.files if f.id != file_id]
        else:
            self._files = [f for f in self._files if f.id != file_id]

    def _list_all_files(self):
        return SimpleNamespace(data=list(self._files))


@pytest.fixture()
def fake_client(monkeypatch):
    store = _FakeVectorStore(
        id="vs1", name="store-1", files=[_FakeFile(id="f1"), _FakeFile(id="f2")]
    )
    orphan = _FakeFile(id="orphan")
    client = _FakeClient([store], [orphan])
    monkeypatch.setattr("corai.vector_storage.cleanup.OpenAI", lambda: client)
    return client


def test_delete_all_vector_stores_handles_files(fake_client):
    _delete_all_vector_stores()
    assert fake_client._stores == []
    assert fake_client._files == []


def test_delete_all_files(monkeypatch):
    files = [_FakeFile(id="keep"), _FakeFile(id="remove")]
    client = _FakeClient([], files)
    monkeypatch.setattr("corai.vector_storage.cleanup.OpenAI", lambda: client)
    _delete_all_files()
    assert client._files == []


def test_delete_all_vector_stores_handles_exceptions(monkeypatch):
    class BrokenClient(_FakeClient):
        def _delete_file(self, *, file_id: str, vector_store_id: str | None = None):
            raise RuntimeError("fail delete")

        def _delete_store(self, store_id: str):
            raise RuntimeError("store fail")

        def _list_all_files(self):
            return SimpleNamespace(data=[_FakeFile(id="orphan")])

        def _delete_file_orphan(self, *, file_id: str):
            raise RuntimeError("orphan fail")

        @property
        def files(self):
            return SimpleNamespace(
                list=self._list_all_files, delete=self._delete_file_orphan
            )

    store = _FakeVectorStore(id="vs1", name="store-1", files=[_FakeFile(id="f1")])
    client = BrokenClient([store], [])
    monkeypatch.setattr("corai.vector_storage.cleanup.OpenAI", lambda: client)

    _delete_all_vector_stores()
