"""Wrapper utilities for managing OpenAI vector stores.

This module provides the VectorStorage class for high-level management of
OpenAI vector stores, including concurrent file uploads, semantic search,
and batch operations.
"""

from __future__ import annotations

import glob
import logging
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

from openai import OpenAI
from openai.pagination import SyncPage
from openai.types.vector_store import VectorStore
from openai.types.vector_store_search_response import VectorStoreSearchResponse
from tqdm import tqdm

from ..settings import OpenAISettings
from ..errors import ConfigurationError, VectorStorageError
from ..types import OpenAIClient
from ..utils import ensure_list, ensure_directory, log
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

    This class provides a high-level interface for managing OpenAI vector stores,
    including file uploads, deletions, and semantic search operations. It handles
    file caching, concurrent uploads, and automatic store creation.

    Parameters
    ----------
    store_name : str
        Name of the vector store to create or connect to.
    client : OpenAIClient or None, optional
        Preconfigured OpenAI-compatible client, by default None.
    model : str or None, optional
        Embedding model identifier. Reads OPENAI_MODEL env var if None,
        by default None.

    Examples
    --------
    Basic usage:

    >>> from openai_sdk_helpers.vector_storage import VectorStorage
    >>> storage = VectorStorage(store_name="documents")
    >>> storage.upload_file("research.pdf")
    >>> results = storage.search("machine learning algorithms", top_k=5)

    Batch file upload:

    >>> patterns = ["docs/*.pdf", "papers/*.txt"]
    >>> stats = storage.upload_files(patterns, overwrite=False)
    >>> print(f"Uploaded {stats.uploaded} files")

    Clean up:

    >>> storage.delete()  # Delete entire store and files

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
        *,
        store_name: str,
        client: OpenAIClient | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the vector store helper.

        Creates or connects to a named vector store using the OpenAI API.
        Requires either a preconfigured client or OPENAI_API_KEY environment
        variable.

        Parameters
        ----------
        store_name : str
            Name of the vector store to create or connect to.
        client : OpenAIClient or None, optional
            Preconfigured OpenAI-compatible client, by default None.
        model : str or None, optional
            Embedding model identifier. Reads OPENAI_MODEL env var if None,
            by default None.

        Raises
        ------
        ConfigurationError
            If no API key or embedding model can be resolved.
        """
        if client is None:
            try:
                settings = OpenAISettings.from_env()
                self._client = settings.create_client()
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc
        else:
            self._client = client

        self._model = model or os.getenv("OPENAI_MODEL")
        if self._model is None:
            raise ConfigurationError("OpenAI model is required")

        self._vector_storage = self._get_or_create_vector_storage(store_name)
        self._existing_files: dict[str, str] | None = {}

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

        Searches for an existing vector store with the specified name. If not
        found, creates a new one.

        Parameters
        ----------
        store_name : str
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
                log(
                    f"Failed to load existing files: {exc}",
                    level=logging.ERROR,
                    exc=exc,
                )
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
            log(
                f"Failed to load existing files: {exc}",
                level=logging.ERROR,
                exc=exc,
            )
            return {}

    def upload_file(
        self,
        file_path: str,
        *,
        purpose: str = "assistants",
        attributes: dict[str, str | float | bool] | None = None,
        overwrite: bool = False,
        refresh_cache: bool = False,
        expires_after: int | None = None,
    ) -> VectorStorageFileInfo:
        """Upload a single file to the vector store.

        Handles text and binary files with automatic encoding detection.
        Skips upload if file already exists unless overwrite is True.

        Parameters
        ----------
        file_path : str
            Local path to the file to upload.
        purpose : str, optional
            Purpose of the file (e.g., "assistants"), by default "assistants".
        attributes : dict[str, str | float | bool] or None, optional
            Custom attributes to associate with the file. The file_name
            attribute is added automatically, by default None.
        overwrite : bool, optional
            When True, re-upload even if a file with the same name exists,
            by default False.
        refresh_cache : bool, optional
            When True, refresh the local cache of existing files before
            checking for duplicates, by default False.
        expires_after : int or None, optional
            Number of seconds after which the file expires and is deleted.
            If None and purpose is "user_data", defaults to 86400 (24 hours).

        Returns
        -------
        VectorStorageFileInfo
            Information about the uploaded file, including its ID and status.
        """
        file_name = os.path.basename(file_path)
        attributes = dict(attributes or {})
        attributes["file_name"] = file_name

        # Default to 24 hours expiration for user_data files
        if expires_after is None and purpose == "user_data":
            expires_after = 86400  # 24 hours in seconds

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
                file=(file_name, file_data),
                purpose=cast(Any, purpose),  # Cast to avoid type error
                expires_after=expires_after,  # type: ignore
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
            log(
                f"Error uploading {file_name}: {str(exc)}",
                level=logging.ERROR,
                exc=exc,
            )
            return VectorStorageFileInfo(
                name=file_name, id="", status="error", error=str(exc)
            )

    def upload_files(
        self,
        file_patterns: str | list[str],
        *,
        purpose: str = "assistants",
        attributes: dict[str, str | float | bool] | None = None,
        overwrite: bool = False,
        expires_after: int | None = None,
    ) -> VectorStorageFileStats:
        """Upload files matching glob patterns using a thread pool.

        Expands glob patterns to find matching files and uploads them
        concurrently using up to 10 worker threads. Shows progress bar
        during upload.

        Parameters
        ----------
        file_patterns : str or list[str]
            Glob pattern or list of patterns (e.g., '/path/**/*.txt').
        purpose : str, optional
            Purpose assigned to uploaded files, by default "assistants".
        attributes : dict[str, str | float | bool] or None, optional
            Custom attributes to associate with each file, by default None.
        overwrite : bool, optional
            When True, re-upload files even if files with the same name
            exist, by default False.
        expires_after : int or None, optional
            Number of seconds after which files expire and are deleted.
            If None and purpose is "user_data", defaults to 86400 (24 hours).

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
                    purpose=purpose,
                    attributes=attributes,
                    overwrite=overwrite,
                    refresh_cache=False,
                    expires_after=expires_after,
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
        """Delete a specific file from the vector store and OpenAI Files API.

        Removes the file from the vector store, then deletes it from OpenAI's
        Files API storage. Updates the local cache. The operation is irreversible.

        Parameters
        ----------
        file_id : str
            Identifier of the file to delete.

        Returns
        -------
        VectorStorageFileInfo
            Information about the deletion operation with status
            "success" or "failed".
        """
        try:
            # First remove from vector store
            self._client.vector_stores.files.delete(
                vector_store_id=self._vector_storage.id, file_id=file_id
            )

            # Then delete the actual file from OpenAI storage
            try:
                self._client.files.delete(file_id)
                log(f"Deleted file {file_id} from OpenAI Files API")
            except Exception as file_delete_exc:
                # Log but don't fail if file doesn't exist or can't be deleted
                log(
                    f"Warning: Could not delete file {file_id} from Files API: {file_delete_exc}",
                    level=logging.WARNING,
                    exc=file_delete_exc,
                )

            to_remove = [k for k, v in self.existing_files.items() if v == file_id]
            for key in to_remove:
                del self.existing_files[key]

            return VectorStorageFileInfo(
                name=to_remove[0] if to_remove else "", id=file_id, status="success"
            )
        except Exception as exc:
            log(
                f"Error deleting file {file_id}: {str(exc)}",
                level=logging.ERROR,
                exc=exc,
            )
            return VectorStorageFileInfo(
                name="", id=file_id, status="failed", error=str(exc)
            )

    def delete_files(self, file_ids: list[str]) -> VectorStorageFileStats:
        """Delete multiple files using a thread pool.

        Performs concurrent deletions using up to 10 worker threads with
        progress tracking. Updates the local cache for each successful
        deletion.

        Parameters
        ----------
        file_ids : list[str]
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

        Removes each file from the vector store and deletes it from OpenAI's
        Files API storage before deleting the store itself. The local cache
        is cleared after deletion.

        Warning: This operation is irreversible and will permanently delete
        the vector store and all its files from OpenAI storage.
        """
        try:
            existing_files = list(self.existing_files.items())
            for file_name, file_id in existing_files:
                log(
                    f"Deleting file {file_id} ({file_name}) from vector store and Files API"
                )
                self.delete_file(file_id)

            self._client.vector_stores.delete(self._vector_storage.id)
            self._existing_files = None  # clear cache
            log(f"Vector store '{self._vector_storage.name}' deleted successfully.")

        except Exception as exc:
            log(
                f"Error deleting vector store '{self._vector_storage.name}': {str(exc)}",
                level=logging.ERROR,
                exc=exc,
            )

    def download_files(self, output_dir: str) -> VectorStorageFileStats:
        """Download every file in the vector store to a local directory.

        Creates the output directory if needed. Uses file names from
        attributes or falls back to file IDs.

        Parameters
        ----------
        output_dir : str
            Destination directory where files will be written. Created if
            it does not exist.

        Returns
        -------
        VectorStorageFileStats
            Aggregated statistics describing the download results.
        """
        ensure_directory(Path(output_dir))

        try:
            files = self._client.vector_stores.files.list(
                vector_store_id=self._vector_storage.id
            )
            store_files = list(getattr(files, "data", files))
        except Exception as exc:
            log(
                f"Failed to list files for download: {exc}",
                level=logging.ERROR,
                exc=exc,
            )
            return VectorStorageFileStats(
                total=0,
                fail=1,
                errors=[
                    VectorStorageFileInfo(
                        name="", id="", status="error", error=str(exc)
                    )
                ],
            )

        stats = VectorStorageFileStats(total=len(store_files))

        for file_ref in store_files:
            file_id = getattr(file_ref, "id", "")
            attributes = getattr(file_ref, "attributes", {}) or {}
            file_name = attributes.get("file_name") or file_id
            target_path = os.path.join(output_dir, file_name)

            try:
                content = self._client.files.content(file_id=file_id)
                if isinstance(content, bytes):
                    data = content
                elif hasattr(content, "read"):
                    data = cast(bytes, content.read())
                else:
                    raise TypeError("Unsupported content type for file download")
                with open(target_path, "wb") as handle:
                    handle.write(data)
                stats.success += 1
            except Exception as exc:
                log(
                    f"Failed to download {file_id}: {exc}",
                    level=logging.ERROR,
                    exc=exc,
                )
                stats.fail += 1
                stats.errors.append(
                    VectorStorageFileInfo(
                        name=file_name, id=file_id, status="error", error=str(exc)
                    )
                )

        return stats

    def search(
        self, query: str, *, top_k: int = 5
    ) -> SyncPage[VectorStoreSearchResponse] | None:
        """Perform a semantic search within the vector store.

        Uses the configured embedding model to find the most relevant
        documents matching the query.

        Parameters
        ----------
        query : str
            Search query string.
        top_k : int, optional
            Maximum number of results to return, by default 5.

        Returns
        -------
        SyncPage[VectorStoreSearchResponse] or None
            Page of search results from the OpenAI API, or None if an
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
            log(
                f"Error searching vector store: {str(exc)}",
                level=logging.ERROR,
                exc=exc,
            )
            return None

    def summarize(self, query: str, *, top_k: int = 15) -> str | None:
        """Perform a semantic search and summarize results by topic.

        Retrieves top search results and generates a summary. This method
        is designed to be overridden in application-specific wrappers.

        Parameters
        ----------
        query : str
            Search query string used for summarization.
        top_k : int, optional
            Number of top search results to retrieve and summarize,
            by default 15.

        Returns
        -------
        str or None
            Summary generated by the OpenAI model, or None when no results
            are available or an error occurs.

        Raises
        ------
        RuntimeError
            If no summarizer is configured (default behavior).
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
