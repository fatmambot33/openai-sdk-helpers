"""Configuration-driven Streamlit chat application.

This module implements a complete Streamlit chat interface that loads its
configuration from a Python module. It handles conversation state, message
rendering, response execution, and resource cleanup.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from openai_sdk_helpers.response import ResponseBase, attach_vector_store
from openai_sdk_helpers.streamlit_app import (
    StreamlitAppConfig,
    _load_configuration,
)
from openai_sdk_helpers.structure.base import StructureBase
from openai_sdk_helpers.utils import (
    coerce_jsonable,
    customJSONEncoder,
    ensure_list,
)

# Supported file extensions for OpenAI Assistants file search and vision
SUPPORTED_FILE_EXTENSIONS = (
    ".gif",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".webp",
)


def _validate_file_type(filename: str) -> bool:
    """Check if a file has a supported extension for upload.

    Supports both document formats (for file search) and image formats
    (for vision analysis).

    Parameters
    ----------
    filename : str
        Name of the file to validate.

    Returns
    -------
    bool
        True if the file extension is supported, False otherwise.
    """
    file_ext = Path(filename).suffix.lower()
    return file_ext in SUPPORTED_FILE_EXTENSIONS


def _cleanup_temp_files(file_paths: list[str] | None = None) -> None:
    """Delete temporary files that were created for uploads.

    Parameters
    ----------
    file_paths : list[str] or None, default None
        Specific file paths to delete. If None, deletes all tracked
        temporary files from session state.

    Notes
    -----
    Silently ignores errors when deleting files that may have already
    been removed or are inaccessible.
    """
    paths_to_delete = file_paths or st.session_state.get("temp_file_paths", [])
    for path in paths_to_delete:
        try:
            if os.path.exists(path):
                os.remove(path)
        except (OSError, IOError):
            pass  # Silently ignore if file already deleted or inaccessible

    if file_paths is None:
        st.session_state["temp_file_paths"] = []


def _extract_assistant_text(response: ResponseBase[Any]) -> str:
    """Extract the latest assistant message as readable text.

    Searches the response's message history for the most recent assistant
    or tool message and extracts displayable text content.

    Parameters
    ----------
    response : ResponseBase[Any]
        Active response session with message history.

    Returns
    -------
    str
        Concatenated assistant text, or empty string if unavailable.

    Examples
    --------
    >>> text = _extract_assistant_text(response)
    >>> print(text)
    """
    message = response.get_last_assistant_message() or response.get_last_tool_message()
    if message is None:
        return ""

    # Check if the message content has output_text attribute
    output_text = getattr(message.content, "output_text", None)
    if output_text:
        return str(output_text)

    content = getattr(message.content, "content", None)
    if content is None:
        return ""

    text_parts: list[str] = []
    for part in ensure_list(content):
        # Handle both dict-like parts and object-like parts
        text_content = None
        if hasattr(part, "text"):
            text_content = getattr(part, "text", None)
        elif isinstance(part, dict):
            text_content = part.get("text")

        if text_content:
            # If text_content is a string, use it directly (dict-style)
            if isinstance(text_content, str):
                text_parts.append(text_content)
            # If text_content is an object with a value attribute, extract that value (object-style)
            else:
                text_value = getattr(text_content, "value", None)
                if text_value:
                    text_parts.append(text_value)
    if text_parts:
        return "\n\n".join(text_parts)
    return ""


def _render_summary(result: Any, response: ResponseBase[Any]) -> str:
    """Generate display text for the chat transcript.

    Converts the response result into a human-readable format suitable
    for display in the Streamlit chat interface. Handles structured
    outputs, dictionaries, and raw text.

    Parameters
    ----------
    result : Any
        Parsed result from ResponseBase.run_sync.
    response : ResponseBase[Any]
        Response instance containing message history.

    Returns
    -------
    str
        Display-ready summary text for the chat transcript.

    Notes
    -----
    Falls back to extracting assistant text from message history if
    the result cannot be formatted directly.
    """
    if isinstance(result, StructureBase):
        return str(result)
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return json.dumps(coerce_jsonable(result), indent=2, cls=customJSONEncoder)
    if result:
        coerced = coerce_jsonable(result)
        try:
            return json.dumps(coerced, indent=2, cls=customJSONEncoder)
        except TypeError:
            return str(result)

    fallback_text = _extract_assistant_text(response)
    if fallback_text:
        return fallback_text
    return "No response returned."


def _build_raw_output(result: Any, response: ResponseBase[Any]) -> dict[str, Any]:
    """Assemble raw JSON payload for the expandable transcript section.

    Creates a structured dictionary containing both the parsed result
    and the complete conversation history for debugging and inspection.

    Parameters
    ----------
    result : Any
        Parsed result from the response execution.
    response : ResponseBase[Any]
        Response session with complete message history.

    Returns
    -------
    dict[str, Any]
        Mapping with 'parsed' data and 'conversation' messages.

    Examples
    --------
    >>> raw = _build_raw_output(result, response)
    >>> raw.keys()
    dict_keys(['parsed', 'conversation'])
    """
    return {
        "parsed": coerce_jsonable(result),
        "conversation": response.messages.to_json(),
    }


def _get_response_instance(configuration: StreamlitAppConfig) -> ResponseBase[Any]:
    """Instantiate and cache the configured ResponseBase.

    Creates a new response instance from the configuration if not already
    cached in session state. Applies vector store attachments and cleanup
    settings based on configuration.

    Parameters
    ----------
    configuration : StreamlitAppConfig
        Loaded configuration with response handler definition.

    Returns
    -------
    ResponseBase[Any]
        Active response instance for the current Streamlit session.

    Raises
    ------
    TypeError
        If the configured response cannot produce a ResponseBase.

    Notes
    -----
    The response instance is cached in st.session_state['response_instance']
    to maintain conversation state across Streamlit reruns.
    """
    if "response_instance" in st.session_state:
        cached = st.session_state["response_instance"]
        if isinstance(cached, ResponseBase):
            return cached

    response = configuration.create_response()

    if configuration.preserve_vector_stores:
        setattr(response, "_cleanup_system_vector_storage", False)
        setattr(response, "_cleanup_user_vector_storage", False)

    vector_stores = configuration.normalized_vector_stores()
    if vector_stores:
        attach_vector_store(response=response, vector_stores=vector_stores)

    st.session_state["response_instance"] = response
    return response


def _reset_chat(close_response: bool = True) -> None:
    """Clear conversation and optionally close the response session.

    Saves the current conversation to disk, closes the response to clean
    up resources, and clears the chat history from session state. Also
    cleans up any temporary files that were created for uploads.

    Parameters
    ----------
    close_response : bool, default True
        Whether to call close() on the cached response instance,
        triggering resource cleanup.

    Notes
    -----
    This function mutates st.session_state in-place, clearing the
    chat_history, response_instance, and temp_file_paths keys.
    """
    response = st.session_state.get("response_instance")
    if close_response and isinstance(response, ResponseBase):
        filepath = f"./data/{response.name}.{response.uuid}.json"
        response.save(filepath)
        response.close()

    # Clean up temporary files
    _cleanup_temp_files()

    st.session_state["chat_history"] = []
    st.session_state.pop("response_instance", None)


def _init_session_state() -> None:
    """Initialize Streamlit session state for chat functionality.

    Creates the chat_history list in session state if it doesn't exist,
    enabling conversation persistence across Streamlit reruns. Also
    initializes a list for tracking temporary file paths that need cleanup.

    Notes
    -----
    This function should be called early in the app lifecycle to ensure
    session state is properly initialized before rendering chat UI.
    """
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "temp_file_paths" not in st.session_state:
        st.session_state["temp_file_paths"] = []
    if "current_attachments" not in st.session_state:
        st.session_state["current_attachments"] = []
    if "attachment_names" not in st.session_state:
        st.session_state["attachment_names"] = []


def _render_chat_history() -> None:
    """Display the conversation transcript from session state.

    Iterates through chat_history in session state and renders each
    message with appropriate formatting. Assistant messages include
    an expandable raw output section.

    Notes
    -----
    Uses Streamlit's chat_message context manager for role-based
    message styling.
    """
    for message in st.session_state.get("chat_history", []):
        role = message.get("role", "assistant")
        with st.chat_message(role):
            if role == "assistant":
                st.markdown(message.get("summary", ""))
                raw_output = message.get("raw")
                if raw_output is not None:
                    with st.expander("Raw output", expanded=False):
                        st.json(raw_output)
            else:
                st.markdown(message.get("content", ""))
                attachments = message.get("attachments", [])
                if attachments:
                    st.caption(
                        f"📎 {len(attachments)} file(s) attached: {', '.join(attachments)}"
                    )


def _handle_user_message(
    prompt: str,
    configuration: StreamlitAppConfig,
    attachment_paths: list[str] | None = None,
    attachment_names: list[str] | None = None,
) -> None:
    """Process user input and generate assistant response.

    Appends the user message to chat history, executes the response
    handler, and adds the assistant's reply to the conversation.
    Handles errors gracefully by displaying them in the UI.

    Parameters
    ----------
    prompt : str
        User-entered text to send to the assistant.
    configuration : StreamlitAppConfig
        Loaded configuration with response handler definition.
    attachment_paths : list[str] or None, default None
        Optional list of file paths to attach to the message.
    attachment_names : list[str] or None, default None
        Optional list of original filenames for display purposes.

    Notes
    -----
    Errors during response execution are caught and displayed in the
    chat transcript rather than crashing the application. The function
    triggers a Streamlit rerun after successful response generation.
    """
    # Use provided display names or fall back to extracting from paths
    display_names = (
        attachment_names
        if attachment_names
        else [Path(p).name for p in attachment_paths] if attachment_paths else []
    )

    st.session_state["chat_history"].append(
        {"role": "user", "content": prompt, "attachments": display_names}
    )
    try:
        response = _get_response_instance(configuration)
    except Exception as exc:  # pragma: no cover - surfaced in UI
        st.error(f"Failed to start response session: {exc}")
        return

    try:
        with st.spinner("Thinking..."):
            result = response.run_sync(content=prompt, files=attachment_paths)
        summary = _render_summary(result, response)
        raw_output = _build_raw_output(result, response)
        st.session_state["chat_history"].append(
            {"role": "assistant", "summary": summary, "raw": raw_output}
        )
        st.rerun()
    except Exception as exc:  # pragma: no cover - surfaced in UI
        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "summary": f"Encountered an error: {exc}",
                "raw": {"error": str(exc)},
            }
        )
        st.error("Something went wrong, but your chat history is still here.")


def main(config_path: Path) -> None:
    """Run the configuration-driven Streamlit chat application.

    Entry point for the Streamlit app that loads configuration, sets up
    the UI, manages session state, and handles user interactions.

    Parameters
    ----------
    config_path : Path
        Filesystem location of the configuration module.

    Notes
    -----
    This function should be called as the entry point for the Streamlit
    application. It handles the complete application lifecycle including
    configuration loading, UI rendering, and chat interactions.

    Examples
    --------
    >>> from pathlib import Path
    >>> main(Path("./my_config.py"))
    """
    configuration = _load_configuration(config_path)
    st.set_page_config(page_title=configuration.display_title, layout="wide")
    _init_session_state()

    st.title(configuration.display_title)
    if configuration.description:
        st.caption(configuration.description)
    if configuration.model:
        st.caption(f"Model: {configuration.model}")

    close_col, _ = st.columns([1, 5])
    with close_col:
        if st.button("Close chat", type="secondary"):
            _reset_chat()
            st.toast("Chat closed.")

    _render_chat_history()

    # File uploader form - auto-clears on submit
    with st.form("file_upload_form", clear_on_submit=True):
        uploaded_files = st.file_uploader(
            "Attach files (optional)",
            accept_multiple_files=True,
            help=f"Supported formats: {', '.join(sorted(SUPPORTED_FILE_EXTENSIONS))}",
        )
        submit_files = st.form_submit_button("Attach files")

    # Process uploaded files if form was submitted
    attachment_paths: list[str] = []
    original_filenames: list[str] = []
    if submit_files and uploaded_files:
        invalid_files = []
        for uploaded_file in uploaded_files:
            if not _validate_file_type(uploaded_file.name):
                invalid_files.append(uploaded_file.name)
                continue

            # Create temporary file with the uploaded content
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded_file.name).suffix
            ) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file.flush()
                attachment_paths.append(tmp_file.name)
                original_filenames.append(uploaded_file.name)
                # Track for cleanup
                if tmp_file.name not in st.session_state.get("temp_file_paths", []):
                    st.session_state["temp_file_paths"].append(tmp_file.name)

        if invalid_files:
            st.warning(
                f"⚠️ Unsupported file types: {', '.join(invalid_files)}. "
                f"Supported: {', '.join(sorted(SUPPORTED_FILE_EXTENSIONS))}"
            )
        if attachment_paths:
            st.session_state["current_attachments"] = attachment_paths
            st.session_state["attachment_names"] = original_filenames
            st.info(f"📎 {len(attachment_paths)} file(s) attached")

    # Get attachment paths from session state if they were previously attached
    attachment_paths = st.session_state.get("current_attachments", [])
    attachment_display_names = st.session_state.get("attachment_names", [])
    if attachment_paths:
        st.caption(f"Ready to send: {', '.join(attachment_display_names)}")

    prompt = st.chat_input("Message the assistant")
    if prompt:
        # Clear attachments before rerun to prevent them from being sent again
        st.session_state["current_attachments"] = []
        st.session_state["attachment_names"] = []
        _handle_user_message(
            prompt,
            configuration,
            attachment_paths or None,
            attachment_display_names or None,
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python app.py <config_path>")
        sys.exit(1)
    config_path = Path(sys.argv[1])
    main(config_path)
