"""Configuration-driven Streamlit chat application.

This module implements a complete Streamlit chat interface that loads its
configuration from a Python module. It handles conversation state, message
rendering, response execution, and resource cleanup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from openai_sdk_helpers.response import BaseResponse, attach_vector_store
from openai_sdk_helpers.streamlit_app import (
    StreamlitAppConfig,
    _load_configuration,
)
from openai_sdk_helpers.structure.base import BaseStructure
from openai_sdk_helpers.utils import coerce_jsonable, ensure_list, log


def _extract_assistant_text(response: BaseResponse[Any]) -> str:
    """Extract the latest assistant message as readable text.

    Searches the response's message history for the most recent assistant
    or tool message and extracts displayable text content.

    Parameters
    ----------
    response : BaseResponse[Any]
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

    content = getattr(message.content, "content", None)
    if content is None:
        return ""

    text_parts: list[str] = []
    for part in ensure_list(content):
        text_value = getattr(getattr(part, "text", None), "value", None)
        if text_value:
            text_parts.append(text_value)
    if text_parts:
        return "\n\n".join(text_parts)
    return ""


def _render_summary(result: Any, response: BaseResponse[Any]) -> str:
    """Generate display text for the chat transcript.

    Converts the response result into a human-readable format suitable
    for display in the Streamlit chat interface. Handles structured
    outputs, dictionaries, and raw text.

    Parameters
    ----------
    result : Any
        Parsed result from BaseResponse.run_sync.
    response : BaseResponse[Any]
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
    if isinstance(result, BaseStructure):
        return result.print()
    if isinstance(result, dict):
        return json.dumps(result, indent=2)
    if result:
        return str(result)

    fallback_text = _extract_assistant_text(response)
    if fallback_text:
        return fallback_text
    return "No response returned."


def _build_raw_output(result: Any, response: BaseResponse[Any]) -> dict[str, Any]:
    """Assemble raw JSON payload for the expandable transcript section.

    Creates a structured dictionary containing both the parsed result
    and the complete conversation history for debugging and inspection.

    Parameters
    ----------
    result : Any
        Parsed result from the response execution.
    response : BaseResponse[Any]
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


def _get_response_instance(config: StreamlitAppConfig) -> BaseResponse[Any]:
    """Instantiate and cache the configured BaseResponse.

    Creates a new response instance from the configuration if not already
    cached in session state. Applies vector store attachments and cleanup
    settings based on configuration.

    Parameters
    ----------
    config : StreamlitAppConfig
        Loaded configuration with response handler definition.

    Returns
    -------
    BaseResponse[Any]
        Active response instance for the current Streamlit session.

    Raises
    ------
    TypeError
        If the configured response cannot produce a BaseResponse.

    Notes
    -----
    The response instance is cached in st.session_state['response_instance']
    to maintain conversation state across Streamlit reruns.
    """
    if "response_instance" in st.session_state:
        cached = st.session_state["response_instance"]
        if isinstance(cached, BaseResponse):
            return cached

    response = config.create_response()

    if config.preserve_vector_stores:
        setattr(response, "_cleanup_system_vector_storage", False)
        setattr(response, "_cleanup_user_vector_storage", False)

    vector_stores = config.normalized_vector_stores()
    if vector_stores:
        attach_vector_store(response=response, vector_stores=vector_stores)

    st.session_state["response_instance"] = response
    return response


def _reset_chat(close_response: bool = True) -> None:
    """Clear conversation and optionally close the response session.

    Saves the current conversation to disk, closes the response to clean
    up resources, and clears the chat history from session state.

    Parameters
    ----------
    close_response : bool, default True
        Whether to call close() on the cached response instance,
        triggering resource cleanup.

    Notes
    -----
    This function mutates st.session_state in-place, clearing the
    chat_history and response_instance keys.
    """
    response = st.session_state.get("response_instance")
    if close_response and isinstance(response, BaseResponse):
        filepath = f"./data/{response.name}.{response.uuid}.json"
        response.save(filepath)
        response.close()
    st.session_state["chat_history"] = []
    st.session_state.pop("response_instance", None)


def _init_session_state() -> None:
    """Initialize Streamlit session state for chat functionality.

    Creates the chat_history list in session state if it doesn't exist,
    enabling conversation persistence across Streamlit reruns.

    Notes
    -----
    This function should be called early in the app lifecycle to ensure
    session state is properly initialized before rendering chat UI.
    """
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


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


def _handle_user_message(prompt: str, config: StreamlitAppConfig) -> None:
    """Process user input and generate assistant response.

    Appends the user message to chat history, executes the response
    handler, and adds the assistant's reply to the conversation.
    Handles errors gracefully by displaying them in the UI.

    Parameters
    ----------
    prompt : str
        User-entered text to send to the assistant.
    config : StreamlitAppConfig
        Loaded configuration with response handler definition.

    Notes
    -----
    Errors during response execution are caught and displayed in the
    chat transcript rather than crashing the application. The function
    triggers a Streamlit rerun after successful response generation.
    """
    st.session_state["chat_history"].append({"role": "user", "content": prompt})
    try:
        response = _get_response_instance(config)
    except Exception as exc:  # pragma: no cover - surfaced in UI
        st.error(f"Failed to start response session: {exc}")
        return

    try:
        with st.spinner("Thinking..."):
            result = response.run_sync(content=prompt)
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
    config = _load_configuration(config_path)
    st.set_page_config(page_title=config.display_title, layout="wide")
    _init_session_state()

    st.title(config.display_title)
    if config.description:
        st.caption(config.description)
    if config.model:
        st.caption(f"Model: {config.model}")

    close_col, _ = st.columns([1, 5])
    with close_col:
        if st.button("Close chat", type="secondary"):
            _reset_chat()
            st.toast("Chat closed.")

    _render_chat_history()

    prompt = st.chat_input("Message the assistant")
    if prompt:
        _handle_user_message(prompt, config)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python app.py <config_path>")
        sys.exit(1)
    config_path = Path(sys.argv[1])
    main(config_path)
