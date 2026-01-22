"""Prompt optimization and configuration helpers for document extraction."""

from __future__ import annotations

import json
from typing import Sequence

from ..agent.base import AgentBase
from ..agent.configuration import AgentConfiguration
from ..response.configuration import ResponseConfiguration
from ..response.prompter import PROMPTER
from ..settings import OpenAISettings
from ..structure.extraction import DocumentExtractorConfig, ExampleData
from ..structure.prompt import PromptStructure

EXTRACTOR_CONFIG_GENERATOR = ResponseConfiguration(
    name="document_extractor_config_generator",
    instructions=(
        "Generate a DocumentExtractorConfig using the provided inputs.\n"
        "Requirements:\n"
        "- Follow the example approach: examples should be high-quality and match the prompt.\n"
        "- Set the configuration name exactly as provided.\n"
        "- Preserve the provided prompt description and extraction classes.\n"
        "- Include all provided examples without modification.\n"
        "- Do not add or remove extraction classes or examples."
    ),
    tools=None,
    input_structure=None,
    output_structure=DocumentExtractorConfig,
    add_output_instructions=True,
)

PROMPT_OPTIMIZER_AGENT_INSTRUCTIONS = (
    "Optimize the extraction prompt for clarity and precision. "
    "Return the optimized prompt as structured output.\n\n"
    f"{PromptStructure.get_prompt()}"
)

EXTRACTOR_CONFIG_AGENT_INSTRUCTIONS = (
    "Generate a DocumentExtractorConfig using the provided details. "
    "Follow the example approach: examples should be high-quality and match the prompt. "
    "Set the configuration name exactly as provided. "
    "Preserve the prompt description, extraction classes, and examples.\n\n"
    f"{DocumentExtractorConfig.get_prompt()}"
)


def _format_extractor_prompt_request(
    prompt: str,
    extraction_classes: Sequence[str],
    additional_context: str | None,
) -> str:
    """Format the prompt-optimization request payload.

    Parameters
    ----------
    prompt : str
        User-provided prompt content.
    extraction_classes : Sequence[str]
        Extraction classes to include.
    additional_context : str or None
        Optional extra context to include.

    Returns
    -------
    str
        Formatted prompt optimization request.
    """
    class_lines = "\n".join(f"- {item}" for item in extraction_classes)
    request_lines = [
        "Optimize the extraction prompt using the details below:",
        f"User prompt: {prompt}",
        "Extraction classes:",
        class_lines or "- None provided",
    ]
    if additional_context:
        request_lines.append(f"Additional context: {additional_context}")
    return "\n".join(request_lines)


def _format_extractor_config_request(
    name: str,
    prompt_description: str,
    extraction_classes: Sequence[str],
    examples: Sequence[ExampleData],
) -> str:
    """Format the extractor configuration request payload.

    Parameters
    ----------
    name : str
        Name for the extractor configuration.
    prompt_description : str
        Optimized prompt description to use.
    extraction_classes : Sequence[str]
        Extraction classes to include.
    examples : Sequence[ExampleData]
        Example payloads to include.

    Returns
    -------
    str
        Formatted configuration request.
    """
    serialized_examples = [example.to_json() for example in examples]
    examples_json = json.dumps(serialized_examples, indent=2)
    class_lines = "\n".join(f"- {item}" for item in extraction_classes)
    return "\n".join(
        [
            "Build a DocumentExtractorConfig using the details below:",
            f"Name: {name}",
            f"Prompt description: {prompt_description}",
            "Extraction classes:",
            class_lines or "- None provided",
            "Examples (JSON):",
            examples_json,
        ]
    )


def optimize_extractor_prompt(
    openai_settings: OpenAISettings,
    prompt: str,
    extraction_classes: Sequence[str],
    *,
    additional_context: str | None = None,
) -> str:
    """Generate an optimized prompt description for extraction.

    Parameters
    ----------
    openai_settings : OpenAISettings
        Settings used to configure the OpenAI client.
    prompt : str
        User-supplied prompt content.
    extraction_classes : Sequence[str]
        Extraction classes to include in the optimized prompt.
    additional_context : str or None, default None
        Optional context that should influence prompt generation.

    Returns
    -------
    str
        Optimized prompt description.

    Raises
    ------
    TypeError
        If the prompter response does not return a prompt string.
    """
    request_text = _format_extractor_prompt_request(
        prompt,
        extraction_classes,
        additional_context,
    )
    response = PROMPTER.gen_response(openai_settings=openai_settings)
    try:
        result = response.run_sync(request_text)
    finally:
        response.close()

    if isinstance(result, PromptStructure):
        return result.prompt
    if isinstance(result, str):
        return result
    raise TypeError("Prompter response must return a PromptStructure or string.")


def optimize_extractor_prompt_with_agent(
    openai_settings: OpenAISettings,
    prompt: str,
    extraction_classes: Sequence[str],
    *,
    additional_context: str | None = None,
) -> str:
    """Generate an optimized prompt description using AgentBase.

    Parameters
    ----------
    openai_settings : OpenAISettings
        Settings used to configure the agent model.
    prompt : str
        User-supplied prompt content.
    extraction_classes : Sequence[str]
        Extraction classes to include in the optimized prompt.
    additional_context : str or None, default None
        Optional context that should influence prompt generation.

    Returns
    -------
    str
        Optimized prompt description.

    Raises
    ------
    TypeError
        If the agent response does not return a prompt string.
    ValueError
        If no default model is configured.
    """
    if not openai_settings.default_model:
        raise ValueError("OpenAISettings.default_model is required for agent runs.")
    request_text = _format_extractor_prompt_request(
        prompt,
        extraction_classes,
        additional_context,
    )
    configuration = AgentConfiguration(
        name="extractor_prompt_optimizer",
        description="Optimize extraction prompt descriptions.",
        model=openai_settings.default_model,
        instructions=PROMPT_OPTIMIZER_AGENT_INSTRUCTIONS,
        output_structure=PromptStructure,
    )
    agent = AgentBase(configuration=configuration)
    result = agent.run_sync(request_text)

    if isinstance(result, PromptStructure):
        return result.prompt
    if isinstance(result, str):
        return result
    raise TypeError("Agent response must return a PromptStructure or string.")


def generate_document_extractor_config(
    openai_settings: OpenAISettings,
    name: str,
    prompt: str,
    extraction_classes: Sequence[str],
    examples: Sequence[ExampleData],
    *,
    additional_context: str | None = None,
) -> DocumentExtractorConfig:
    """Generate a DocumentExtractorConfig using response-based helpers.

    Parameters
    ----------
    openai_settings : OpenAISettings
        Settings used to configure the OpenAI client.
    name : str
        Name for the extractor configuration.
    prompt : str
        User-supplied prompt content.
    extraction_classes : Sequence[str]
        Extraction classes to include in the configuration.
    examples : Sequence[ExampleData]
        Example payloads supplied to LangExtract.
    additional_context : str or None, default None
        Optional context that should influence prompt generation.

    Returns
    -------
    DocumentExtractorConfig
        Generated extractor configuration.

    Raises
    ------
    TypeError
        If the generator response does not return a DocumentExtractorConfig.
    ValueError
        If no examples are provided.
    """
    if not examples:
        raise ValueError("At least one ExampleData instance is required.")
    prompt_description = optimize_extractor_prompt(
        openai_settings,
        prompt,
        extraction_classes,
        additional_context=additional_context,
    )
    request_text = _format_extractor_config_request(
        name,
        prompt_description,
        extraction_classes,
        examples,
    )
    response = EXTRACTOR_CONFIG_GENERATOR.gen_response(openai_settings=openai_settings)
    try:
        result = response.run_sync(request_text)
    finally:
        response.close()

    if isinstance(result, DocumentExtractorConfig):
        return result
    if isinstance(result, dict):
        return DocumentExtractorConfig.model_validate(result)
    raise TypeError(
        "Extractor config generator must return a DocumentExtractorConfig or dict."
    )


def generate_document_extractor_config_with_agent(
    openai_settings: OpenAISettings,
    name: str,
    prompt: str,
    extraction_classes: Sequence[str],
    examples: Sequence[ExampleData],
    *,
    additional_context: str | None = None,
) -> DocumentExtractorConfig:
    """Generate a DocumentExtractorConfig using AgentBase workflows.

    Parameters
    ----------
    openai_settings : OpenAISettings
        Settings used to configure the agent model.
    name : str
        Name for the extractor configuration.
    prompt : str
        User-supplied prompt content.
    extraction_classes : Sequence[str]
        Extraction classes to include in the configuration.
    examples : Sequence[ExampleData]
        Example payloads supplied to LangExtract.
    additional_context : str or None, default None
        Optional context that should influence prompt generation.

    Returns
    -------
    DocumentExtractorConfig
        Generated extractor configuration.

    Raises
    ------
    TypeError
        If the agent response does not return a DocumentExtractorConfig.
    ValueError
        If no examples are provided.
    """
    if not examples:
        raise ValueError("At least one ExampleData instance is required.")
    if not openai_settings.default_model:
        raise ValueError("OpenAISettings.default_model is required for agent runs.")
    prompt_description = optimize_extractor_prompt_with_agent(
        openai_settings,
        prompt,
        extraction_classes,
        additional_context=additional_context,
    )
    request_text = _format_extractor_config_request(
        name,
        prompt_description,
        extraction_classes,
        examples,
    )
    configuration = AgentConfiguration(
        name="extractor_config_generator",
        description="Generate DocumentExtractorConfig instances.",
        model=openai_settings.default_model,
        instructions=EXTRACTOR_CONFIG_AGENT_INSTRUCTIONS,
        output_structure=DocumentExtractorConfig,
    )
    agent = AgentBase(configuration=configuration)
    result = agent.run_sync(request_text)

    if isinstance(result, DocumentExtractorConfig):
        return result
    if isinstance(result, dict):
        return DocumentExtractorConfig.model_validate(result)
    raise TypeError(
        "Agent config generator must return a DocumentExtractorConfig or dict."
    )


__all__ = [
    "EXTRACTOR_CONFIG_GENERATOR",
    "EXTRACTOR_CONFIG_AGENT_INSTRUCTIONS",
    "PROMPT_OPTIMIZER_AGENT_INSTRUCTIONS",
    "generate_document_extractor_config",
    "generate_document_extractor_config_with_agent",
    "optimize_extractor_prompt",
    "optimize_extractor_prompt_with_agent",
]
