"""Cleanup utilities for vector stores.

This module provides destructive operations for cleaning up OpenAI vector
stores and files. Use these functions with caution as they perform
irreversible deletions.
"""

from __future__ import annotations

import logging

from openai import OpenAI

from ..utils import log


def _delete_all_vector_stores() -> None:
    """Delete all vector stores and clean up any orphaned files.

    Iterates over every vector store in the account, deleting all files
    within each store before removing the store itself. After all stores
    are deleted, removes any remaining orphaned files.

    Warning: This operation is irreversible and will delete all vector
    stores and files in the account.

    Examples
    --------
    >>> from openai_sdk_helpers.vector_storage import _delete_all_vector_stores
    >>> _delete_all_vector_stores()  # doctest: +SKIP
    """
    try:
        client = OpenAI()
        stores = client.vector_stores.list().data
        log(f"Found {len(stores)} vector stores.")

        attached_file_ids = set()

        for store in stores:
            log(f"Deleting vector store: {store.name} (ID: {store.id})")

            files = client.vector_stores.files.list(vector_store_id=store.id).data
            for file in files:
                attached_file_ids.add(file.id)
                log(f" - Deleting file {file.id}")
                try:
                    client.vector_stores.files.delete(
                        vector_store_id=store.id, file_id=file.id
                    )
                except Exception as file_err:
                    log(
                        f"Failed to delete file {file.id}: {file_err}",
                        level=logging.WARNING,
                    )

            try:
                client.vector_stores.delete(store.id)
                log(f"Vector store {store.name} deleted.")
            except Exception as store_err:
                log(
                    f"Failed to delete vector store {store.name}: {store_err}",
                    level=logging.WARNING,
                )

        log("Checking for orphaned files in client.files...")
        all_files = client.files.list().data
        for file in all_files:
            if file.id not in attached_file_ids:
                try:
                    log(f"Deleting orphaned file {file.id}")
                    client.files.delete(file_id=file.id)
                except Exception as exc:
                    log(
                        f"Failed to delete orphaned file {file.id}: {exc}",
                        level=logging.WARNING,
                        exc=exc,
                    )

    except Exception as exc:
        log(f"Error during cleanup: {exc}", level=logging.ERROR, exc=exc)


def _delete_all_files() -> None:
    """Delete all files from the OpenAI account.

    Iterates over every file in the account and deletes them without
    checking vector store associations. Use with extreme caution.

    Warning: This operation is irreversible and will delete all files
    in the account, regardless of their usage in vector stores.

    Examples
    --------
    >>> from openai_sdk_helpers.vector_storage import _delete_all_files
    >>> _delete_all_files()  # doctest: +SKIP
    """
    client = OpenAI()
    all_files = client.files.list().data
    for file in all_files:
        try:
            log(f"Deleting file {file.id}")
            client.files.delete(file_id=file.id)
        except Exception as exc:
            log(
                f"Failed to delete file {file.id}: {exc}",
                level=logging.WARNING,
                exc=exc,
            )
