"""Comprehensive OpenAI Files API wrapper.

This module provides a complete, professional implementation of the OpenAI Files API
with automatic file tracking, lifecycle management, and cleanup capabilities.

References
----------
OpenAI Files API: https://platform.openai.com/docs/api-reference/files
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, BinaryIO, Literal, Sequence, cast

from openai import OpenAI, NOT_GIVEN
from openai.types import FileDeleted, FileObject
from openai.pagination import SyncCursorPage

from .utils import log

# Valid purposes for file uploads
FilePurpose = Literal[
    "assistants",
    "batch",
    "fine-tune",
    "user_data",
    "vision",
]


class FilesAPIManager:
    """Comprehensive manager for OpenAI Files API operations.

    Provides full access to the OpenAI Files API with automatic file tracking,
    lifecycle management, and cleanup capabilities. Tracks all uploaded files
    and ensures proper deletion on cleanup.

    Parameters
    ----------
    client : OpenAI
        An initialized OpenAI client instance used for making API calls.
    auto_track : bool, default=True
        If True, automatically tracks all files uploaded through the `create`
        method, making them eligible for automatic deletion via `cleanup()`.

    Attributes
    ----------
    tracked_files : dict[str, FileObject]
        Dictionary of tracked file IDs to FileObject instances.

    Methods
    -------
    create(file, purpose)
        Upload a file to OpenAI Files API.
    retrieve(file_id)
        Retrieve information about a specific file.
    list(purpose, limit)
        List files, optionally filtered by purpose.
    delete(file_id)
        Delete a specific file.
    retrieve_content(file_id)
        Download file content.
    batch_upload(files, purpose, track, expires_after)
        Upload multiple files to the Files API.
    cleanup()
        Delete all tracked files.

    Examples
    --------
    >>> from openai import OpenAI
    >>> from openai_sdk_helpers.files_api import FilesAPIManager
    >>>
    >>> client = OpenAI()
    >>> files_manager = FilesAPIManager(client)
    >>>
    >>> # Upload a file
    >>> with open("document.pdf", "rb") as f:
    ...     file_obj = files_manager.create(f, purpose="user_data")
    >>>
    >>> # List all user data files
    >>> user_files = files_manager.list(purpose="user_data")
    >>>
    >>> # Retrieve file content
    >>> content = files_manager.retrieve_content(file_obj.id)
    >>>
    >>> # Clean up all tracked files
    >>> files_manager.cleanup()
    """

    def __init__(self, client: OpenAI, auto_track: bool = True):
        """Initialize the Files API manager.

        Parameters
        ----------
        client : OpenAI
            OpenAI client instance.
        auto_track : bool, default True
            Automatically track uploaded files for cleanup.

        Raises
        ------
        ValueError
            If the client is not a valid OpenAI client.

        Examples
        --------
        >>> from openai import OpenAI
        >>> client = OpenAI()
        >>> manager = FilesAPIManager(client)
        """
        self._client = client
        self._auto_track = auto_track
        self.tracked_files: dict[str, FileObject] = {}

    def create(
        self,
        file: BinaryIO | Path | str,
        purpose: FilePurpose,
        track: bool | None = None,
        expires_after: int | None = None,
    ) -> FileObject:
        """Upload a file to the OpenAI Files API.

        Parameters
        ----------
        file : BinaryIO, Path, or str
            File-like object, path to file, or file path string.
        purpose : FilePurpose
            The intended purpose of the uploaded file.
            Options: "assistants", "batch", "fine-tune", "user_data", "vision"
        track : bool or None, default None
            Override auto_track for this file. If None, uses instance setting.
        expires_after : int or None, default None
            Number of seconds after which the file expires and is deleted.
            If None and purpose is "user_data", defaults to 86400 (24 hours).
            For other purposes, files don't expire unless explicitly set.

        Returns
        -------
        FileObject
            Information about the uploaded file.

        Raises
        ------
        FileNotFoundError
            If file path doesn't exist.
        ValueError
            If purpose is invalid.

        Examples
        --------
        >>> # Upload from file path (user_data expires in 24h by default)
        >>> file_obj = manager.create("data.jsonl", purpose="user_data")
        >>>
        >>> # Upload with custom expiration (1 hour)
        >>> file_obj = manager.create("temp.txt", purpose="user_data", expires_after=3600)
        >>>
        >>> # Upload from file handle
        >>> with open("image.png", "rb") as f:
        ...     file_obj = manager.create(f, purpose="vision")
        >>>
        >>> # Upload without tracking
        >>> file_obj = manager.create("temp.txt", purpose="user_data", track=False)
        """
        should_track = track if track is not None else self._auto_track

        # Default to 24 hours expiration for user_data files
        if expires_after is None and purpose == "user_data":
            expires_after = 86400  # 24 hours in seconds

        # Handle different file input types
        # Prepare expires_after in OpenAI API format if provided
        expires_after_param = None
        if expires_after is not None:
            expires_after_param = cast(
                Any, {"anchor": "created_at", "seconds": expires_after}
            )

        if isinstance(file, (Path, str)):
            file_path = Path(file).resolve()
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file}")

            # Use only the basename as filename (remove path)
            filename = file_path.name
            with open(file_path, "rb") as f:
                # Pass tuple (filename, file_data) to set custom filename
                if expires_after_param is not None:
                    file_obj = self._client.files.create(
                        file=(filename, f),
                        purpose=purpose,
                        expires_after=expires_after_param,
                    )
                else:
                    file_obj = self._client.files.create(
                        file=(filename, f), purpose=purpose
                    )
        else:
            # Assume it's a BinaryIO
            if expires_after_param is not None:
                file_obj = self._client.files.create(
                    file=file,
                    purpose=purpose,
                    expires_after=expires_after_param,
                )
            else:
                file_obj = self._client.files.create(file=file, purpose=purpose)

        if should_track:
            self.tracked_files[file_obj.id] = file_obj
            expiry_msg = f" (expires in {expires_after}s)" if expires_after else ""
            log(
                f"Uploaded and tracking file {file_obj.id} ({file_obj.filename}) "
                f"with purpose '{purpose}'{expiry_msg}"
            )
        else:
            log(
                f"Uploaded file {file_obj.id} ({file_obj.filename}) "
                f"with purpose '{purpose}' (not tracked)"
            )

        return file_obj

    def retrieve(self, file_id: str) -> FileObject:
        """Retrieve information about a specific file.

        Parameters
        ----------
        file_id : str
            The ID of the file to retrieve.

        Returns
        -------
        FileObject
            Information about the file.

        Raises
        ------
        NotFoundError
            If the file ID does not exist.

        Examples
        --------
        >>> file_info = manager.retrieve("file-abc123")
        >>> print(f"Filename: {file_info.filename}")
        >>> print(f"Size: {file_info.bytes} bytes")
        """
        return self._client.files.retrieve(file_id)

    def list(
        self,
        purpose: FilePurpose | None = None,
        limit: int | None = None,
    ) -> SyncCursorPage[FileObject]:
        """List files, optionally filtered by purpose.

        Parameters
        ----------
        purpose : FilePurpose or None, default None
            Filter files by purpose. If None, returns all files.
        limit : int or None, default None
            Maximum number of files to return. If None, returns all.

        Returns
        -------
        SyncCursorPage[FileObject]
            Page of file objects matching the criteria.

        Raises
        ------
        APIError
            If the OpenAI API call fails.

        Examples
        --------
        >>> # List all files
        >>> all_files = manager.list()
        >>>
        >>> # List user data files
        >>> user_files = manager.list(purpose="user_data")
        >>>
        >>> # List up to 10 files
        >>> recent_files = manager.list(limit=10)
        """
        limit_param = NOT_GIVEN if limit is None else limit
        if purpose is not None:
            return self._client.files.list(
                purpose=purpose, limit=cast(Any, limit_param)
            )
        return self._client.files.list(limit=cast(Any, limit_param))

    def delete(self, file_id: str, untrack: bool = True) -> FileDeleted:
        """Delete a specific file from OpenAI Files API.

        Parameters
        ----------
        file_id : str
            The ID of the file to delete.
        untrack : bool, default True
            Remove from tracked files after deletion.

        Returns
        -------
        FileDeleted
            Confirmation of file deletion.

        Raises
        ------
        NotFoundError
            If the file ID does not exist.

        Examples
        --------
        >>> result = manager.delete("file-abc123")
        >>> print(f"Deleted: {result.deleted}")
        """
        result = self._client.files.delete(file_id)

        if untrack and file_id in self.tracked_files:
            del self.tracked_files[file_id]
            log(f"Deleted and untracked file {file_id}")
        else:
            log(f"Deleted file {file_id}")

        return result

    def retrieve_content(self, file_id: str) -> bytes:
        """Download and retrieve the content of a file.

        Parameters
        ----------
        file_id : str
            The ID of the file to download.

        Returns
        -------
        bytes
            The raw bytes of the file content.

        Raises
        ------
        NotFoundError
            If the file ID does not exist.

        Examples
        --------
        >>> content = manager.retrieve_content("file-abc123")
        >>> with open("downloaded.pdf", "wb") as f:
        ...     f.write(content)
        """
        return self._client.files.content(file_id).read()

    def batch_upload(
        self,
        files: Sequence[BinaryIO | Path | str],
        purpose: FilePurpose,
        track: bool | None = None,
        expires_after: int | None = None,
    ) -> list[FileObject]:
        """Upload multiple files to the OpenAI Files API.

        Parameters
        ----------
        files : Sequence[BinaryIO | Path | str]
            File-like objects or file paths to upload.
        purpose : FilePurpose
            The intended purpose of the uploaded files.
        track : bool or None, default None
            Override auto_track for these uploads. If None, uses instance setting.
        expires_after : int or None, default None
            Number of seconds after which files expire. See ``create`` for details.

        Returns
        -------
        list[FileObject]
            Uploaded file objects in the same order as ``files``.

        Examples
        --------
        >>> files = ["doc1.pdf", "doc2.pdf"]
        >>> uploaded = manager.batch_upload(files, purpose="user_data")
        >>> [file.id for file in uploaded]
        """
        if not files:
            return []
        return [
            self.create(
                file_path,
                purpose=purpose,
                track=track,
                expires_after=expires_after,
            )
            for file_path in files
        ]

    def cleanup(self) -> dict[str, bool]:
        """Delete all tracked files.

        Returns
        -------
        dict[str, bool]
            Dictionary mapping file IDs to deletion success status.

        Examples
        --------
        >>> results = manager.cleanup()
        >>> print(f"Deleted {sum(results.values())} files")
        """
        results = {}
        file_ids = list(self.tracked_files.keys())

        for file_id in file_ids:
            try:
                self.delete(file_id, untrack=True)
                results[file_id] = True
            except Exception as exc:
                log(
                    f"Error deleting tracked file {file_id}: {exc}",
                    level=logging.WARNING,
                    exc=exc,
                )
                results[file_id] = False

        if results:
            successful = sum(results.values())
            log(f"Cleanup complete: {successful}/{len(results)} files deleted")
        else:
            log("No tracked files to clean up")

        return results

    def __enter__(self) -> FilesAPIManager:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with automatic cleanup."""
        self.cleanup()

    def __len__(self) -> int:
        """Return number of tracked files."""
        return len(self.tracked_files)

    def __repr__(self) -> str:
        """Return string representation of the manager."""
        return f"FilesAPIManager(tracked_files={len(self.tracked_files)})"


__all__ = ["FilesAPIManager", "FilePurpose"]
