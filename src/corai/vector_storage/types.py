"""Type definitions for vector storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VectorStorageFileInfo:
    """Information about a file stored in a vector store.

    Attributes
    ----------
    name : str
        File name associated with the vector store item.
    id : str
        Unique identifier of the file in the vector store.
    status : str
        Outcome of the operation (for example ``"success"`` or ``"error"``).
    error : str, optional
        Error message when the operation fails. Default ``None``.

    Methods
    -------
    None
    """

    name: str
    id: str
    status: str
    error: Optional[str] = None


@dataclass
class VectorStorageFileStats:
    """Aggregate statistics about batch file operations.

    Attributes
    ----------
    total : int
        Total number of files processed.
    success : int
        Number of files successfully handled.
    fail : int
        Number of files that failed to process.
    errors : list[VectorStorageFileInfo]
        Details for each failed file.

    Methods
    -------
    None
    """

    total: int = 0
    success: int = 0
    fail: int = 0
    errors: List[VectorStorageFileInfo] = field(default_factory=list)
