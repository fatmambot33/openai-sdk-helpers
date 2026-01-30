"""File attachment helpers for the Agents SDK."""

from __future__ import annotations

from typing import Any, Literal

from ..files_api import FilePurpose, FilesAPIManager
from ..settings import OpenAISettings
from ..utils import create_image_data_url, ensure_list, is_image_file


def build_agent_input_messages(
    content: str | list[str],
    files: str | list[str] | None = None,
    *,
    files_manager: FilesAPIManager | None = None,
    openai_settings: OpenAISettings | None = None,
    file_purpose: FilePurpose = "user_data",
    image_detail: Literal["low", "high", "auto"] = "auto",
) -> list[dict[str, Any]]:
    """Build Agents SDK input messages with file attachments.

    Parameters
    ----------
    content : str or list[str]
        Prompt text or list of prompt texts to send.
    files : str, list[str], or None, default None
        Optional file path or list of file paths. Image files are sent as
        base64-encoded ``input_image`` entries. Document files are uploaded
        using ``files_manager`` and sent as ``input_file`` entries.
    files_manager : FilesAPIManager or None, default None
        File upload helper used to create file IDs for document uploads.
        Required when ``files`` contains non-image documents.
    openai_settings : OpenAISettings or None, default None
        Optional OpenAI settings used to build a FilesAPIManager when one is
        not provided. When supplied, ``openai_settings.create_client()`` is
        used to initialize the Files API manager.
    file_purpose : FilePurpose, default "user_data"
        Purpose passed to the Files API when uploading document files.
    image_detail : {"low", "high", "auto"}, default "auto"
        Detail hint passed along with base64-encoded image inputs.

    Returns
    -------
    list[dict[str, Any]]
        Agents SDK input messages that include text and optional file entries.

    Raises
    ------
    ValueError
        If document files are provided without a ``files_manager``.

    Examples
    --------
    >>> from openai import OpenAI
    >>> from openai_sdk_helpers.files_api import FilesAPIManager
    >>> from openai_sdk_helpers.agent.files import build_agent_input_messages
    >>> client = OpenAI()
    >>> files_manager = FilesAPIManager(client)
    >>> messages = build_agent_input_messages(
    ...     "Summarize this document",
    ...     files="report.pdf",
    ...     files_manager=files_manager,
    ... )
    """
    contents = ensure_list(content)
    all_files = ensure_list(files)

    image_files: list[str] = []
    document_files: list[str] = []
    for file_path in all_files:
        if is_image_file(file_path):
            image_files.append(file_path)
        else:
            document_files.append(file_path)

    attachments: list[dict[str, Any]] = []

    if document_files:
        if files_manager is None and openai_settings is not None:
            files_manager = FilesAPIManager(openai_settings.create_client())
        if files_manager is None:
            raise ValueError(
                "files_manager is required to upload document files for agent input."
            )
        expires_after = 86400 if file_purpose == "user_data" else None
        if hasattr(files_manager, "batch_upload"):
            uploaded_files = files_manager.batch_upload(
                document_files,
                purpose=file_purpose,
                expires_after=expires_after,
            )
        else:
            uploaded_files = [
                files_manager.create(
                    file_path, purpose=file_purpose, expires_after=expires_after
                )
                for file_path in document_files
            ]
        for uploaded_file in uploaded_files:
            attachments.append({"type": "input_file", "file_id": uploaded_file.id})

    for image_path in image_files:
        image_url, detail = create_image_data_url(image_path, detail=image_detail)
        attachments.append(
            {"type": "input_image", "image_url": image_url, "detail": detail}
        )

    messages: list[dict[str, Any]] = []
    for index, raw_content in enumerate(contents):
        text = raw_content.strip()
        content_items: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
        if index == 0:
            content_items.extend(attachments)
        messages.append({"role": "user", "content": content_items})

    return messages


__all__ = ["build_agent_input_messages"]
