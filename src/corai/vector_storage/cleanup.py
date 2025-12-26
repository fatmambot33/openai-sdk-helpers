"""Cleanup helpers for vector stores."""

from __future__ import annotations

import logging

from openai import OpenAI

from ..utils import log


def _delete_all_vector_stores() -> None:
    """Delete all vector stores and clean up any orphaned files.

    This utility iterates over every vector store owned by the account,
    deleting each one after removing all of its files. Any standalone files that
    remain after the stores are deleted are also removed.

    Returns
    -------
    None
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
                    )

    except Exception as exc:
        log(f"Error during cleanup: {exc}", level=logging.ERROR)


def _delete_all_files() -> None:
    """Delete all files from the OpenAI account.

    This utility iterates over every file owned by the account and deletes them.
    It does not check for vector stores, so it will delete all files regardless
    of their association.

    Returns
    -------
    None
    """
    client = OpenAI()
    all_files = client.files.list().data
    for file in all_files:
        try:
            log(f"Deleting file {file.id}")
            client.files.delete(file_id=file.id)
        except Exception as exc:
            log(f"Failed to delete file {file.id}: {exc}", level=logging.WARNING)
