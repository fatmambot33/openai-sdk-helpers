"""Response-native translation helpers."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from ..prompt import PromptRenderer
from ..settings import OpenAISettings
from ..structure import TranslationStructure


class TranslatorResponse:
    """Translate content through the OpenAI Responses API.

    Parameters
    ----------
    model : str
        Model identifier used for translation requests.
    temperature : float | None, default=0
        Sampling temperature used for translation requests.
    template_path : str | Path | None, default=None
        Optional custom prompt template path. Uses ``translator.jinja`` when None.
    data_path : Path | None, default=None
        Optional path reserved for compatibility with response helpers.

    Methods
    -------
    run_async(content, target_language, context)
        Translate content asynchronously into a target language.
    run_sync(content, target_language, context)
        Translate content synchronously into a target language.
    """

    def __init__(
        self,
        *,
        model: str,
        temperature: float | None = 0,
        template_path: str | Path | None = None,
        data_path: Path | None = None,
    ) -> None:
        """Initialize response-mode translator configuration.

        Parameters
        ----------
        model : str
            Model identifier used for translation requests.
        temperature : float | None, default=0
            Sampling temperature used for translation requests.
        template_path : str | Path | None, default=None
            Optional custom prompt template path. Uses ``translator.jinja`` when None.
        data_path : Path | None, default=None
            Optional path reserved for compatibility with response helpers.
        """
        self._model = model
        self._temperature = temperature
        self._template_path = str(template_path) if template_path is not None else None
        self._data_path = data_path
        self._renderer = PromptRenderer()
        self._client: Any | None = None

    async def run_async(
        self,
        content: str,
        *,
        target_language: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TranslationStructure:
        """Translate content asynchronously.

        Parameters
        ----------
        content : str
            Source text to translate.
        target_language : str
            Language to translate the content into.
        context : dict[str, Any] or None, default=None
            Optional context values merged into prompt rendering.

        Returns
        -------
        TranslationStructure
            Structured translation output.
        """
        prompt_context: Dict[str, Any] = {"target_language": target_language}
        if context:
            prompt_context.update(context)

        template_name = self._template_path or "translator.jinja"
        prompt = self._renderer.render(template_name, context=prompt_context)

        payload = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_text",
                        "text": f"Target language: {target_language}",
                    },
                    {"type": "input_text", "text": f"Text to translate:\n{content}"},
                ],
            }
        ]

        response = await asyncio.to_thread(
            self._get_client().responses.create,
            model=self._model,
            input=payload,
            text=TranslationStructure.response_format(),
            temperature=self._temperature,
        )

        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise RuntimeError("No structured output returned from Responses API")

        parsed = json.loads(output_text)
        return TranslationStructure.from_json(parsed)

    def run_sync(
        self,
        content: str,
        *,
        target_language: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> TranslationStructure:
        """Translate content synchronously.

        Parameters
        ----------
        content : str
            Source text to translate.
        target_language : str
            Language to translate the content into.
        context : dict[str, Any] or None, default=None
            Optional context values merged into prompt rendering.

        Returns
        -------
        TranslationStructure
            Structured translation output.
        """

        async def runner() -> TranslationStructure:
            return await self.run_async(
                content,
                target_language=target_language,
                context=context,
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(runner())

        result: TranslationStructure | None = None
        error: Exception | None = None

        def _thread_func() -> None:
            nonlocal result, error
            try:
                result = asyncio.run(runner())
            except Exception as exc:  # pragma: no cover - defensive branch.
                error = exc

        thread = threading.Thread(target=_thread_func)
        thread.start()
        thread.join()

        if error is not None:
            raise error
        if result is None:
            raise RuntimeError("Translation did not return a result")
        return result

    def _get_client(self) -> Any:
        """Return a cached OpenAI client.

        Returns
        -------
        Any
            OpenAI client instance.
        """
        if self._client is not None:
            return self._client
        openai_settings = OpenAISettings.from_env(default_model=self._model)
        self._client = openai_settings.create_client()
        return self._client


def translate_response(
    *,
    content: str,
    target_language: str,
    model: str,
    temperature: float | None = 0,
    context: Optional[Dict[str, Any]] = None,
    template_path: str | Path | None = None,
    data_path: Path | None = None,
) -> TranslationStructure:
    """Translate text via the Responses API.

    Parameters
    ----------
    content : str
        Source text to translate.
    target_language : str
        Language to translate the content into.
    model : str
        Model identifier used for translation requests.
    temperature : float | None, default=0
        Sampling temperature used for translation requests.
    context : dict[str, Any] or None, default=None
        Optional context values merged into prompt rendering.
    template_path : str | Path | None, default=None
        Optional custom prompt template path. Uses ``translator.jinja`` when None.
    data_path : Path | None, default=None
        Optional path reserved for compatibility with response helpers.

    Returns
    -------
    TranslationStructure
        Structured translation output.
    """
    translator = TranslatorResponse(
        model=model,
        temperature=temperature,
        template_path=template_path,
        data_path=data_path,
    )
    return translator.run_sync(
        content,
        target_language=target_language,
        context=context,
    )
