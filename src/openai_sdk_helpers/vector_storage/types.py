"""Type definitions for vector storage.

This module defines data structures for tracking file operations and
statistics when working with OpenAI vector stores.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VectorStorageFileInfo:
    """Information about a file stored in a vector store.

    Tracks the status and details of a single file operation within a
    vector store, including upload, download, or deletion outcomes.

    Attributes
    ----------
    name : str
        File name associated with the vector store item.
    id : str
        Unique identifier of the file in the vector store.
    status : str
        Outcome of the operation (e.g., "success", "error", "existing").
    error : str or None, optional
        Error message when the operation fails, by default None.
    """

    name: str
    id: str
    status: str
    error: str | None = None


@dataclass
class VectorStorageFileStats:
    """Aggregate statistics about batch file operations.

    Tracks outcomes for batch upload, download, or deletion operations
    across multiple files in a vector store.

    Attributes
    ----------
    total : int
        Total number of files processed, by default 0.
    success : int
        Number of files successfully handled, by default 0.
    fail : int
        Number of files that failed to process, by default 0.
    errors : list[VectorStorageFileInfo]
        Details for each failed file, by default empty list.
    """

    total: int = 0
    success: int = 0
    fail: int = 0
    errors: list[VectorStorageFileInfo] = field(default_factory=list)
