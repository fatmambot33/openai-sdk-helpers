"""Streamlit chat application driven by a developer configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st  # type: ignore[import-not-found]

from openai_sdk_helpers.response.base import ResponseBase
from openai_sdk_helpers.response.vector_store import attach_vector_store
from openai_sdk_helpers.streamlit_app import (
    DEFAULT_CONFIG_PATH,
    StreamlitAppConfig,
    _load_configuration,
)
from openai_sdk_helpers.structure.base import BaseStructure
from openai_sdk_helpers.utils import ensure_list
from openai_sdk_helpers.utils.core import coerce_jsonable


def _extract_assistant_text(response: ResponseBase[Any]) -> str:
    """Return the latest assistant message as a friendly string.

    Parameters
    ----------
    response : ResponseBase[Any]
        Active response session containing message history.

    Returns
    -------
    str
        Concatenated assistant text, or an empty string when unavailable.
    """
    message = response.get_last_message(role="assistant")
    if message is None:
        return ""

    content = getattr(message.content, "content", None)
    if content is None:
        return ""

    text_parts: List[str] = []
    for part in ensure_list(content):
        text_value = getattr(getattr(part, "text", None), "value", None)
        if text_value:
            text_parts.append(text_value)
    if text_parts:
        return "\n\n".join(text_parts)
    return ""


def _render_summary(result: Any, response: ResponseBase[Any]) -> str:
    """Generate the assistant-facing summary shown in the transcript.

    Parameters
    ----------
    result : Any
        Parsed result returned from ``ResponseBase.run_sync``.
    response : ResponseBase[Any]
        Response instance containing the latest assistant message.

    Returns
    -------
    str
        Display-ready summary text for the chat transcript.
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


def _build_raw_output(result: Any, response: ResponseBase[Any]) -> Dict[str, Any]:
    """Assemble the raw payload shown under the expandable transcript section.

    Parameters
    ----------
    result : Any
        Parsed result returned from the response instance.
    response : ResponseBase[Any]
        Response session containing message history.

    Returns
    -------
    dict[str, Any]
        Mapping that includes parsed data and raw conversation messages.
    """
    return {
        "parsed": coerce_jsonable(result),
        "conversation": response.messages.to_json(),
    }


def _get_response_instance(config: StreamlitAppConfig) -> ResponseBase[Any]:
    """Instantiate and cache the configured :class:`ResponseBase`.

    Parameters
    ----------
    config : StreamlitAppConfig
        Loaded configuration containing the response definition.

    Returns
    -------
    ResponseBase[Any]
        Active response instance for the current session.

    Raises
    ------
    TypeError
        If the configured ``response`` cannot produce ``ResponseBase``.
    """
    if "response_instance" in st.session_state:
        cached = st.session_state["response_instance"]
        if isinstance(cached, ResponseBase):
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
    """Clear the conversation and optionally close the response session.

    Parameters
    ----------
    close_response : bool, default=True
        Whether to call ``close`` on the cached response instance.

    Returns
    -------
    None
        This function mutates ``st.session_state`` in-place.
    """
    response = st.session_state.get("response_instance")
    if close_response and isinstance(response, ResponseBase):
        response.close()
    st.session_state["chat_history"] = []
    st.session_state.pop("response_instance", None)


def _init_session_state() -> None:
    """Prepare Streamlit session state containers.

    Returns
    -------
    None
        This function initializes chat-related session keys when absent.
    """
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []


def _render_chat_history() -> None:
    """Display the conversation transcript from session state.

    Returns
    -------
    None
        Renders chat messages in the current Streamlit session.
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
    """Append a user prompt and stream the assistant reply into the transcript.

    Parameters
    ----------
    prompt : str
        User-entered text to send to the assistant.
    config : StreamlitAppConfig
        Loaded configuration containing the response definition.
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
    except Exception as exc:  # pragma: no cover - surfaced in UI
        st.session_state["chat_history"].append(
            {
                "role": "assistant",
                "summary": f"Encountered an error: {exc}",
                "raw": {"error": str(exc)},
            }
        )
        st.error("Something went wrong, but your chat history is still here.")


def main(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Run the config-driven Streamlit chat app.

    Parameters
    ----------
    config_path : Path, default=DEFAULT_CONFIG_PATH
        Filesystem location of the configuration module.
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
    main()
