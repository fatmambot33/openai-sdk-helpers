"""Wrapper utilities for managing OpenAI vector stores."""

from __future__ import annotations

import glob
import logging
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union

from openai import OpenAI
from openai.pagination import SyncPage
from openai.types.vector_store import VectorStore
from openai.types.vector_store_search_response import VectorStoreSearchResponse
from tqdm import tqdm

from ..utils import ensure_list, log
from .types import VectorStorageFileInfo, VectorStorageFileStats

TEXT_MIME_PREFIXES = ("text/",)
ALLOWED_TEXT_MIME_TYPES = {
    "text/x-c",
    "text/x-c++",
    "text/x-csharp",
    "text/css",
    "text/x-golang",
    "text/html",
    "text/x-java",
    "text/javascript",
    "application/json",
    "text/markdown",
    "text/x-python",
    "text/x-script.python",
    "text/x-ruby",
    "application/x-sh",
    "text/x-tex",
    "application/typescript",
    "text/plain",
}


class VectorStorage:
    """Manage an OpenAI vector store.

    Methods
    -------
    id()
        Return the ID of the underlying vector store.
    existing_files()
        Map cached file names to their IDs.
    upload_file(file_path, purpose, attributes, overwrite, refresh_cache)
        Upload a single file to the vector store.
    upload_files(file_patterns, purpose, attributes, overwrite)
        Upload files matching glob patterns by using a thread pool.
    delete_file(file_id)
        Delete a specific file from the vector store.
    delete_files(file_ids)
        Delete multiple files from the vector store.
    delete()
        Delete the entire vector store and associated files.
    search(query, top_k)
        Perform a search within the vector store.
    summarize(query, top_k)
        Summarize top search results returned by the vector store.
    """

    def __init__(
        self,
        store_name: str,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initialize the vector store helper.

        Parameters
        ----------
        store_name
            Name of the vector store to create or connect to.
        client
            Optional preconfigured ``OpenAI`` client. Default ``None``.
        model
            Embedding model identifier. Default ``None`` to read ``OPENAI_MODEL``.

        Raises
        ------
        ValueError
            If no API key or embedding model can be resolved.
        RuntimeError
            If the OpenAI client cannot be initialized.
        """
        if client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key is None:
                raise ValueError("OpenAI API key is required")
            try:
                self._client = OpenAI(api_key=api_key)
            except Exception as exc:
                raise RuntimeError("Failed to initialize OpenAI client") from exc
        else:
            self._client = client

        self._model = model or os.getenv("OPENAI_MODEL")
        if self._model is None:
            raise ValueError("OpenAI model is required")

        self._vector_storage = self._get_or_create_vector_storage(store_name)
        self._existing_files: Optional[dict[str, str]] = {}

    @property
    def id(self) -> str:
        """Return the ID of the underlying OpenAI ``VectorStore`` object.

        Returns
        -------
        str
            Identifier of the vector store.
        """
        return self._vector_storage.id

    def _get_or_create_vector_storage(self, store_name: str) -> VectorStore:
        """Retrieve an existing vector store or create one if it does not exist.

        Parameters
        ----------
        store_name
            Desired name of the vector store.

        Returns
        -------
        VectorStore
            Retrieved or newly created vector store object.
        """
        vector_stores = self._client.vector_stores.list().data
        existing = next((vs for vs in vector_stores if vs.name == store_name), None)
        return existing or self._client.vector_stores.create(name=store_name)

    @property
    def existing_files(self) -> dict[str, str]:
        """Map file names to their IDs for files currently in the vector store.

        This property lazily loads the file list from the OpenAI API on first
        access and caches it. The cache can be refreshed by calling
        ``_load_existing_files`` or by setting ``refresh_cache=True`` in
        ``upload_file``.

        Returns
        -------
        dict[str, str]
            Mapping of file names to file IDs.
        """
        if self._existing_files is None:
            try:
                files = self._client.vector_stores.files.list(
                    vector_store_id=self._vector_storage.id
                )
                self._existing_files = {}
                for f in files:
                    file_name = (f.attributes or {}).get("file_name")
                    if isinstance(file_name, str) and f.id:
                        self._existing_files[file_name] = f.id

            except Exception as exc:
                log(f"Failed to load existing files: {exc}", level=logging.ERROR)
                self._existing_files = {}
        return self._existing_files

    def _load_existing_files(self) -> dict[str, str]:
        """Force a reload of the existing files from the OpenAI API.

        Returns
        -------
        dict[str, str]
            Updated mapping of file names to file IDs.
        """
        try:
            files = self._client.vector_stores.files.list(
                vector_store_id=self._vector_storage.id
            )
            result: dict[str, str] = {}
            for f in files:
                file_name = (f.attributes or {}).get("file_name")
                if isinstance(file_name, str) and f.id:
                    result[file_name] = f.id
            return result
        except Exception as exc:
            log(f"Failed to load existing files: {exc}", level=logging.ERROR)
            return {}

    def upload_file(
        self,
        file_path: str,
        purpose: str = "assistants",
        attributes: Optional[dict[str, str | float | bool]] = None,
        overwrite: bool = False,
        refresh_cache: bool = False,
    ) -> VectorStorageFileInfo:
        """Upload a single file to the vector store.

        Parameters
        ----------
        file_path
            Local path to the file to upload.
        purpose
            Purpose of the file (for example ``"assistants"``). Default
            ``"assistants"``.
        attributes
            Custom attributes to associate with the file. The ``file_name``
            attribute is added automatically. Default ``None``.
        overwrite
            When ``True``, re-upload even if a file with the same name already
            exists. Default ``False``.
        refresh_cache
            When ``True``, refresh the local cache of existing files before
            checking for duplicates. Default ``False``.

        Returns
        -------
        VectorStorageFileInfo
            Information about the uploaded file, including its ID and status.
        """
        file_name = os.path.basename(file_path)
        attributes = dict(attributes or {})
        attributes["file_name"] = file_name

        if refresh_cache:
            self._existing_files = self._load_existing_files()

        if not overwrite and file_name in self.existing_files:
            return VectorStorageFileInfo(
                name=file_name, id=self.existing_files[file_name], status="existing"
            )

        try:
            mime_type, _ = mimetypes.guess_type(file_path)
            mime_type = mime_type or ""

            if mime_type in ALLOWED_TEXT_MIME_TYPES or mime_type.startswith(
                TEXT_MIME_PREFIXES
            ):
                try:
                    with open(file_path, "r", encoding="utf-8") as handle:
                        content = handle.read()
                    file_data = content.encode("utf-8")
                except UnicodeDecodeError:
                    with open(file_path, "r", encoding="utf-16") as handle:
                        content = handle.read()
                    file_data = content.encode("utf-16")
            else:
                with open(file_path, "rb") as handle:
                    file_data = handle.read()

            file = self._client.files.create(
                file=(file_path, file_data), purpose=purpose  # type: ignore
            )

            self._client.vector_stores.files.create(
                self._vector_storage.id,
                file_id=file.id,
                attributes=attributes,
            )

            self._client.vector_stores.files.poll(
                file.id, vector_store_id=self._vector_storage.id
            )

            self.existing_files[file_name] = file.id

            return VectorStorageFileInfo(name=file_name, id=file.id, status="success")
        except Exception as exc:
            log(f"Error uploading {file_name}: {str(exc)}", level=logging.ERROR)
            return VectorStorageFileInfo(
                name=file_name, id="", status="error", error=str(exc)
            )

    def upload_files(
        self,
        file_patterns: Union[str, List[str]],
        purpose: str = "assistants",
        attributes: Optional[dict[str, str | float | bool]] = None,
        overwrite: bool = False,
    ) -> VectorStorageFileStats:
        """Upload files matching glob patterns to the vector store using a thread pool.

        Parameters
        ----------
        file_patterns
            Glob pattern or list of patterns (for example
            ``'/path/to/files/**/*.txt'``).
        purpose
            Purpose assigned to uploaded files. Default ``"assistants"``.
        attributes
            Custom attributes to associate with each file. Default ``None``.
        overwrite
            When ``True``, re-upload files even if files with the same name
            already exist. Default ``False``.

        Returns
        -------
        VectorStorageFileStats
            Aggregated statistics describing the upload results.
        """
        file_patterns = ensure_list(file_patterns)

        all_paths = set()
        for pattern in file_patterns:
            all_paths.update(glob.glob(pattern, recursive=True))
        if not all_paths:
            log("No files to upload.", level=logging.INFO)
            return VectorStorageFileStats(total=0)

        if not overwrite:
            existing_files = self.existing_files
            all_paths = [
                f for f in all_paths if os.path.basename(f) not in existing_files
            ]

        if not all_paths:
            log("No new files to upload.", level=logging.INFO)
            return VectorStorageFileStats()

        stats = VectorStorageFileStats(total=len(all_paths))

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(
                    self.upload_file,
                    path,
                    purpose,
                    attributes,
                    overwrite,
                    False,
                ): path
                for path in all_paths
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result.status in {"success", "existing"}:
                    stats.success += 1
                    if result.status == "success":
                        self.existing_files[result.name] = result.id
                else:
                    stats.fail += 1
                    stats.errors.append(result)

        return stats

    def delete_file(self, file_id: str) -> VectorStorageFileInfo:
        """Delete a specific file from the vector store.

        Parameters
        ----------
        file_id
            Identifier of the file to delete.

        Returns
        -------
        VectorStorageFileInfo
            Information about the deletion operation with status
            ``"success"`` or ``"failed"``.
        """
        try:
            self._client.vector_stores.files.delete(
                vector_store_id=self._vector_storage.id, file_id=file_id
            )

            to_remove = [k for k, v in self.existing_files.items() if v == file_id]
            for key in to_remove:
                del self.existing_files[key]

            return VectorStorageFileInfo(
                name=to_remove[0] if to_remove else "", id=file_id, status="success"
            )
        except Exception as exc:
            log(f"Error deleting file {file_id}: {str(exc)}", level=logging.ERROR)
            return VectorStorageFileInfo(
                name="", id=file_id, status="failed", error=str(exc)
            )

    def delete_files(self, file_ids: List[str]) -> VectorStorageFileStats:
        """Delete multiple files from the vector store using a thread pool.

        Parameters
        ----------
        file_ids
            List of file IDs to delete.

        Returns
        -------
        VectorStorageFileStats
            Aggregated statistics describing the deletion results.
        """
        total_files = len(file_ids)
        log(f"{total_files} files to delete...")
        stats = VectorStorageFileStats(total=total_files)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(self.delete_file, file_id): file_id
                for file_id in file_ids
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result.status == "success":
                    stats.success += 1
                else:
                    stats.fail += 1
                    stats.errors.append(result)

        return stats

    def delete(self) -> None:
        """Delete the entire vector store and all associated files.

        This operation is irreversible. It first attempts to delete each file
        individually from the store (and updates the local cache) before
        deleting the store itself.

        Returns
        -------
        None
        """
        try:
            existing_files = list(self.existing_files.items())
            for file_name, file_id in existing_files:
                log(f"Deleting file {file_id} ({file_name}) from vector store")
                self.delete_file(file_id)

            self._client.vector_stores.delete(self._vector_storage.id)
            self._existing_files = None  # clear cache
            log(f"Vector store '{self._vector_storage.name}' deleted successfully.")

        except Exception as exc:
            log(
                f"Error deleting vector store '{self._vector_storage.name}': {str(exc)}",
                level=logging.ERROR,
            )

    def search(
        self, query: str, top_k: int = 5
    ) -> Optional[SyncPage[VectorStoreSearchResponse]]:
        """Perform a search within the vector store.

        Parameters
        ----------
        query
            Search query string.
        top_k
            Maximum number of results to return. Default ``5``.

        Returns
        -------
        Optional[SyncPage[VectorStoreSearchResponse]]
            Page of search results from the OpenAI API, or ``None`` if an
            error occurs.
        """
        try:
            response = self._client.vector_stores.search(
                vector_store_id=self._vector_storage.id,
                query=query,
                max_num_results=top_k,
            )
            return response
        except Exception as exc:
            log(f"Error searching vector store: {str(exc)}", level=logging.ERROR)
            return None

    def summarize(self, query: str, top_k: int = 15) -> Optional[str]:
        """Perform a semantic search and summarize results by topic.

        Parameters
        ----------
        query
            Search query string used for summarization.
        top_k
            Number of top search results to retrieve and summarize. Default ``15``.

        Returns
        -------
        Optional[str]
            Summary generated by the OpenAI model or ``None`` when no results
            are available or an error occurs.

        Raises
        ------
        RuntimeError
            If no summarizer is configured for this core helper.
        """
        response = self.search(query, top_k=top_k)
        if not response or not response.data:
            log("No search results to summarize.", level=logging.WARNING)
            return None

        raise RuntimeError(
            "Summarizer is application-specific; override this method in an "
            "application wrapper."
        )


__all__ = ["VectorStorage"]
