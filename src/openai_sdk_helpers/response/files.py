"""File attachment utilities for responses.

This module provides functions for processing file attachments, automatically
detecting file types (images vs documents), and preparing them for the OpenAI API
with appropriate encoding (base64 or vector store). Supports both individual and
batch file processing.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from openai.types.responses.response_input_file_content_param import (
    ResponseInputFileContentParam,
)
from openai.types.responses.response_input_file_param import ResponseInputFileParam
from openai.types.responses.response_input_image_content_param import (
    ResponseInputImageContentParam,
)

from ..utils import create_file_data_url, create_image_data_url, is_image_file, log

if TYPE_CHECKING:  # pragma: no cover
    from .base import ResponseBase


def process_files(
    response: ResponseBase[Any],
    files: list[str],
    use_vector_store: bool = False,
    batch_size: int = 10,
    max_workers: int = 5,
) -> tuple[
    list[ResponseInputFileParam],
    list[ResponseInputFileContentParam],
    list[ResponseInputImageContentParam],
]:
    """Process file attachments and prepare them for OpenAI API.

    Automatically categorizes files by type (images vs documents) and
    processes them appropriately. Supports concurrent processing for efficient
    handling of multiple files.

    Parameters
    ----------
    response : ResponseBase[Any]
        Response instance that will use the processed files.
    files : list[str]
        List of file paths to process.
    use_vector_store : bool, default False
        If True, non-image files are uploaded to a vector store for
        RAG-enabled file search instead of inline base64 encoding.
    batch_size : int, default 10
        Maximum number of files to submit to thread pool at once.
        Processes files in chunks to avoid overwhelming the executor.
    max_workers : int, default 5
        Maximum number of concurrent workers for processing.

    Returns
    -------
    tuple[list, list, list]
        Three lists containing:
        1. Vector store file references (ResponseInputFileParam)
        2. Base64-encoded file content (ResponseInputFileContentParam)
        3. Base64-encoded image content (ResponseInputImageContentParam)

    Notes
    -----
    Inline ``input_file`` attachments only support PDF documents. For other
    document formats, use ``use_vector_store=True`` or convert to PDF before
    calling this helper.

    Examples
    --------
    >>> from openai_sdk_helpers.response import process_files
    >>> vector_files, base64_files, images = process_files(
    ...     response,
    ...     ["photo.jpg", "document.pdf"],
    ...     use_vector_store=False
    ... )

    >>> # Batch process many files
    >>> vector_files, base64_files, images = process_files(
    ...     response,
    ...     ["file1.pdf", "file2.pdf", ...],  # Many files
    ...     batch_size=20,
    ...     max_workers=10
    ... )
    """
    # Categorize files by type
    image_files: list[str] = []
    document_files: list[str] = []

    for file_path in files:
        if is_image_file(file_path):
            image_files.append(file_path)
        else:
            document_files.append(file_path)

    if document_files and not use_vector_store:
        _validate_inline_document_files(document_files)

    # Handle document files (vector store or base64)
    vector_file_refs: list[ResponseInputFileParam] = []
    base64_files: list[ResponseInputFileContentParam] = []

    if document_files:
        if use_vector_store:
            # Upload to vector store (sequential for now)
            vector_file_refs = _upload_to_vector_store(response, document_files)
        else:
            # Use batch processing for base64 encoding
            base64_files = _encode_documents_base64_batch(
                document_files, batch_size, max_workers
            )

    # Handle images (always base64) with batch processing
    image_contents = _encode_images_base64_batch(image_files, batch_size, max_workers)

    return vector_file_refs, base64_files, image_contents


def _validate_inline_document_files(document_files: list[str]) -> None:
    """Validate document files for inline ``input_file`` usage.

    Parameters
    ----------
    document_files : list[str]
        Document file paths that will be sent as inline ``input_file``
        attachments.

    Raises
    ------
    ValueError
        If any document file is not a PDF.
    """
    unsupported_files = [
        file_path
        for file_path in document_files
        if Path(file_path).suffix.lower() != ".pdf"
    ]
    if unsupported_files:
        filenames = ", ".join(Path(path).name for path in unsupported_files)
        raise ValueError(
            "Inline input_file attachments support PDFs only. "
            f"Unsupported files: {filenames}. "
            "Convert to PDF or set use_vector_store=True."
        )


def _upload_to_vector_store(
    response: ResponseBase[Any], document_files: list[str]
) -> list[ResponseInputFileParam]:
    """Upload documents to vector store and return file references.

    Uploads user files with purpose="user_data" for proper categorization
    and cleanup according to OpenAI Files API conventions.

    Parameters
    ----------
    response : ResponseBase[Any]
        Response instance with vector storage.
    document_files : list[str]
        List of document file paths to upload.

    Returns
    -------
    list[ResponseInputFileParam]
        List of file references for vector store files.

    Notes
    -----
    Files are uploaded with purpose="user_data" to distinguish them
    from assistant files. All user files are automatically deleted
    when the response is closed via the vector store cleanup.
    """
    file_refs: list[ResponseInputFileParam] = []

    if response._user_vector_storage is None:
        from openai_sdk_helpers.vector_storage import VectorStorage

        store_name = f"{response.__class__.__name__.lower()}_{response._name}_{response.uuid}_user"
        response._user_vector_storage = VectorStorage(
            store_name=store_name,
            client=response._client,
            model=response._model,
        )
        user_vector_storage = cast(Any, response._user_vector_storage)
        if response._tools is None:
            response._tools = []
        if not any(tool.get("type") == "file_search" for tool in response._tools):
            response._tools.append(
                {
                    "type": "file_search",
                    "vector_store_ids": [user_vector_storage.id],
                }
            )

    user_vector_storage = cast(Any, response._user_vector_storage)
    for file_path in document_files:
        # Upload with purpose="user_data" for user-uploaded files
        uploaded_file = user_vector_storage.upload_file(file_path, purpose="user_data")
        file_refs.append(
            ResponseInputFileParam(type="input_file", file_id=uploaded_file.id)
        )

        # Best-effort tracking with FilesAPIManager (if available on the response)
        files_manager = getattr(response, "_files_manager", None)
        if files_manager is not None:
            # Prefer tracking by file ID; fall back to full object if needed.
            try:
                files_manager.track_file(uploaded_file.id)
            except AttributeError:
                try:
                    files_manager.track_file(uploaded_file)
                except AttributeError:
                    # If the manager does not support tracking in either form,
                    # silently skip to avoid breaking existing behavior.
                    pass
    return file_refs


def _encode_documents_base64(
    document_files: list[str],
) -> list[ResponseInputFileContentParam]:
    """Encode documents as base64 for inline attachment.

    Parameters
    ----------
    document_files : list[str]
        List of document file paths to encode.

    Returns
    -------
    list[ResponseInputFileContentParam]
        List of base64-encoded file content parameters.
    """
    base64_files: list[ResponseInputFileContentParam] = []

    for file_path in document_files:
        file_data_url = create_file_data_url(file_path)
        filename = Path(file_path).name
        base64_files.append(
            ResponseInputFileContentParam(
                type="input_file",
                file_data=file_data_url,
                filename=filename,
            )
        )

    return base64_files


def _encode_documents_base64_batch(
    document_files: list[str],
    batch_size: int = 10,
    max_workers: int = 5,
) -> list[ResponseInputFileContentParam]:
    """Encode documents as base64 with batch processing.

    Uses thread pool for concurrent encoding of multiple files.

    Parameters
    ----------
    document_files : list[str]
        List of document file paths to encode.
    batch_size : int, default 10
        Number of files to process in each batch.
    max_workers : int, default 5
        Maximum number of concurrent workers.

    Returns
    -------
    list[ResponseInputFileContentParam]
        List of base64-encoded file content parameters.
    """
    if not document_files:
        return []

    # If small number of files, process sequentially
    if len(document_files) <= 3:
        return _encode_documents_base64(document_files)

    base64_files: list[ResponseInputFileContentParam] = []

    def encode_single_document(file_path: str) -> ResponseInputFileContentParam:
        """Encode a single document file."""
        file_data_url = create_file_data_url(file_path)
        filename = Path(file_path).name
        return ResponseInputFileContentParam(
            type="input_file",
            file_data=file_data_url,
            filename=filename,
        )

    # Process files concurrently in batches using thread pool
    log(
        f"Processing {len(document_files)} documents in batches of {batch_size} "
        f"with {max_workers} workers"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process files in batches to avoid overwhelming the executor
        for batch_start in range(0, len(document_files), batch_size):
            batch_end = min(batch_start + batch_size, len(document_files))
            batch = document_files[batch_start:batch_end]

            # Submit this batch of tasks
            future_to_file = {
                executor.submit(encode_single_document, file_path): file_path
                for file_path in batch
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                try:
                    result = future.result()
                    base64_files.append(result)
                except Exception as exc:
                    file_path = future_to_file[future]
                    log(f"Error encoding document {file_path}: {exc}", exc=exc)
                    raise

    return base64_files


def _encode_images_base64(
    image_files: list[str],
) -> list[ResponseInputImageContentParam]:
    """Encode images as base64 for inline attachment.

    Parameters
    ----------
    image_files : list[str]
        List of image file paths to encode.

    Returns
    -------
    list[ResponseInputImageContentParam]
        List of base64-encoded image content parameters.
    """
    image_contents: list[ResponseInputImageContentParam] = []

    for image_path in image_files:
        image_url, detail = create_image_data_url(image_path, detail="auto")
        image_contents.append(
            ResponseInputImageContentParam(
                type="input_image",
                image_url=image_url,
                detail=detail,
            )
        )

    return image_contents


def _encode_images_base64_batch(
    image_files: list[str],
    batch_size: int = 10,
    max_workers: int = 5,
) -> list[ResponseInputImageContentParam]:
    """Encode images as base64 with batch processing.

    Uses thread pool for concurrent encoding of multiple images.

    Parameters
    ----------
    image_files : list[str]
        List of image file paths to encode.
    batch_size : int, default 10
        Number of images to process in each batch.
    max_workers : int, default 5
        Maximum number of concurrent workers.

    Returns
    -------
    list[ResponseInputImageContentParam]
        List of base64-encoded image content parameters.
    """
    if not image_files:
        return []

    # If small number of files, process sequentially
    if len(image_files) <= 3:
        return _encode_images_base64(image_files)

    image_contents: list[ResponseInputImageContentParam] = []

    def encode_single_image(image_path: str) -> ResponseInputImageContentParam:
        """Encode a single image file."""
        image_url, detail = create_image_data_url(image_path, detail="auto")
        return ResponseInputImageContentParam(
            type="input_image",
            image_url=image_url,
            detail=detail,
        )

    # Process images concurrently in batches using thread pool
    log(
        f"Processing {len(image_files)} images in batches of {batch_size} "
        f"with {max_workers} workers"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Process images in batches to avoid overwhelming the executor
        for batch_start in range(0, len(image_files), batch_size):
            batch_end = min(batch_start + batch_size, len(image_files))
            batch = image_files[batch_start:batch_end]

            # Submit this batch of tasks
            future_to_file = {
                executor.submit(encode_single_image, image_path): image_path
                for image_path in batch
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                try:
                    result = future.result()
                    image_contents.append(result)
                except Exception as exc:
                    image_path = future_to_file[future]
                    log(f"Error encoding image {image_path}: {exc}", exc=exc)
                    raise

    return image_contents


__all__ = ["process_files"]
