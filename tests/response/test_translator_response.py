from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openai_sdk_helpers.response.translator import (
    TranslatorResponse,
    translate_response,
)
from openai_sdk_helpers.structure import TranslationStructure


def _response_with_output(payload: dict[str, object]) -> SimpleNamespace:
    """Return a fake Responses API payload container."""
    return SimpleNamespace(output_text=json.dumps(payload))


def test_translator_response_translates_content() -> None:
    """TranslatorResponse should parse structured translation output."""
    translator = TranslatorResponse(model="gpt-4o-mini")

    fake_client = MagicMock()
    fake_client.responses.create.return_value = _response_with_output(
        {"text": "Hola mundo"}
    )

    with patch.object(translator, "_get_client", return_value=fake_client):
        result = translator.run_sync("Hello world", target_language="Spanish")

    assert result.text == "Hola mundo"


def test_translator_response_forwards_temperature() -> None:
    """TranslatorResponse should forward temperature to the API call."""
    translator = TranslatorResponse(model="gpt-4o-mini", temperature=0.2)

    fake_client = MagicMock()
    fake_client.responses.create.return_value = _response_with_output(
        {"text": "Bonjour"}
    )

    with patch.object(translator, "_get_client", return_value=fake_client):
        translator.run_sync("Hi", target_language="French")

    _, kwargs = fake_client.responses.create.call_args
    assert kwargs["temperature"] == 0.2


def test_translate_response_builder() -> None:
    """Helper should build TranslatorResponse and execute synchronously."""
    with patch(
        "openai_sdk_helpers.response.translator.TranslatorResponse.run_sync"
    ) as mock_run:
        mock_run.return_value = TranslationStructure(text="Ciao")

        result = translate_response(
            content="Hello",
            target_language="Italian",
            model="gpt-4o-mini",
            temperature=0,
        )

    assert result.text == "Ciao"
    mock_run.assert_called_once_with(
        "Hello",
        target_language="Italian",
        context=None,
    )
