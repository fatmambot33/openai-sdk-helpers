"""Vector store management and utilities.

This module provides high-level interfaces for managing OpenAI vector stores,
including file upload, deletion, search operations, and cleanup utilities.

Classes
-------
VectorStorage
    Manage an OpenAI vector store with file operations and search.
VectorStorageFileInfo
    Information about a single file in a vector store.
VectorStorageFileStats
    Aggregate statistics for batch file operations.

Functions
---------
_delete_all_vector_stores
    Delete all vector stores and clean up orphaned files.
_delete_all_files
    Delete all files from the OpenAI account.
"""

from __future__ import annotations

from .cleanup import _delete_all_files, _delete_all_vector_stores
from .storage import VectorStorage
from .types import VectorStorageFileInfo, VectorStorageFileStats

__all__ = [
    "VectorStorage",
    "VectorStorageFileInfo",
    "VectorStorageFileStats",
    "_delete_all_vector_stores",
    "_delete_all_files",
]
