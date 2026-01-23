"""Prompt optimization and configuration helpers for document extraction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..agent.base import AgentBase
from ..agent.configuration import AgentConfiguration
from ..prompt import PromptRenderer
from ..response.configuration import ResponseConfiguration
from ..response.prompter import PROMPTER
from ..settings import OpenAISettings
from ..structure.extraction import DocumentExtractorConfig, ExampleDataStructure
from ..structure.prompt import PromptStructure

EXTRACTOR_CONFIG_TEMPLATE_NAME = "extractor_config_generator.jinja"
EXTRACTOR_CONFIG_AGENT_INSTRUCTIONS_TEMPLATE = (
    "extractor_config_agent_instructions.jinja"
)
EXTRACTOR_CONFIG_GENERATOR_INSTRUCTIONS_TEMPLATE = (
    "extractor_config_generator_instructions.jinja"
)
EXTRACTOR_PROMPT_OPTIMIZER_INSTRUCTIONS_TEMPLATE = (
    "extractor_prompt_optimizer_agent_instructions.jinja"
)
EXTRACTOR_PROMPT_OPTIMIZER_REQUEST_TEMPLATE = "extractor_prompt_optimizer_request.jinja"
PROMPT_RENDERER = PromptRenderer()

DEFAULT_EXAMPLE_COUNT = 3


def _render_prompt_template(
    template_name: str,
    context: dict[str, object] | None = None,
) -> str:
    """Render a prompt template from the prompt directory.

    Parameters
    ----------
    template_name : str
        Prompt template file name.
    context : dict[str, object] or None, default None
        Context variables for template rendering.

    Returns
    -------
    str
        Rendered prompt content.
    """
    return PROMPT_RENDERER.render(template_name, context=context or {})


EXTRACTOR_CONFIG_GENERATOR = ResponseConfiguration(
    name="document_extractor_config_generator",
    instructions=_render_prompt_template(
        EXTRACTOR_CONFIG_GENERATOR_INSTRUCTIONS_TEMPLATE
    ),
    tools=None,
    input_structure=None,
    output_structure=DocumentExtractorConfig,
    add_output_instructions=True,
)

PROMPT_OPTIMIZER_AGENT_INSTRUCTIONS = _render_prompt_template(
    EXTRACTOR_PROMPT_OPTIMIZER_INSTRUCTIONS_TEMPLATE,
    context={"prompt_schema": PromptStructure.get_prompt()},
)

EXTRACTOR_CONFIG_AGENT_INSTRUCTIONS = _render_prompt_template(
    EXTRACTOR_CONFIG_AGENT_INSTRUCTIONS_TEMPLATE,
    context={"config_schema": DocumentExtractorConfig.get_prompt()},
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
    return _render_prompt_template(
        EXTRACTOR_PROMPT_OPTIMIZER_REQUEST_TEMPLATE,
        context={
            "prompt": prompt,
            "extraction_classes": list(extraction_classes),
            "additional_context": additional_context,
        },
    )


def _format_extractor_config_request(
    name: str,
    prompt_description: str,
    extraction_classes: Sequence[str],
    *,
    example_files: Sequence[str | Path] | None = None,
    example_count: int = DEFAULT_EXAMPLE_COUNT,
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
    example_files : Sequence[str or Path] or None, default None
        Optional file paths to ground the generated examples.
    example_count : int, default 3
        Number of examples to generate.

    Returns
    -------
    str
        Formatted configuration request.
    """
    return PROMPT_RENDERER.render(
        EXTRACTOR_CONFIG_TEMPLATE_NAME,
        context={
            "name": name,
            "prompt_description": prompt_description,
            "extraction_classes": list(extraction_classes),
            "example_count": example_count,
            "example_files": _load_example_files(example_files),
            "examples_json": "- None provided. You must generate examples.",
            "example_requirements": [
                f"Generate {example_count} high-quality examples that align with the prompt.",
                "Ensure each example includes realistic source text and extractions.",
                "Cover every extraction class across the examples.",
            ],
        },
    )


def _format_extractor_config_request_with_examples(
    name: str,
    prompt_description: str,
    extraction_classes: Sequence[str],
    examples: Sequence[ExampleDataStructure],
) -> str:
    """Format the extractor configuration request payload with examples.

    Parameters
    ----------
    name : str
        Name for the extractor configuration.
    prompt_description : str
        Optimized prompt description to use.
    extraction_classes : Sequence[str]
        Extraction classes to include.
    examples : Sequence[ExampleDataStructure]
        Example payloads to include.

    Returns
    -------
    str
        Formatted configuration request.
    """
    serialized_examples = [example.to_json() for example in examples]
    examples_json = json.dumps(serialized_examples, indent=2)
    return PROMPT_RENDERER.render(
        EXTRACTOR_CONFIG_TEMPLATE_NAME,
        context={
            "name": name,
            "prompt_description": prompt_description,
            "extraction_classes": list(extraction_classes),
            "example_count": DEFAULT_EXAMPLE_COUNT,
            "example_files": [],
            "examples_json": examples_json,
            "example_requirements": ["Use the provided examples exactly as written."],
        },
    )


def _load_example_files(
    example_files: Sequence[str | Path] | None,
) -> list[dict[str, str]]:
    """Load optional example files for grounded extraction generation.

    Parameters
    ----------
    example_files : Sequence[str or Path] or None
        File paths to load for grounding examples.

    Returns
    -------
    list of dict[str, str]
        Loaded file metadata including path and content.

    Raises
    ------
    FileNotFoundError
        If any provided file does not exist.
    """
    if not example_files:
        return []
    loaded_files: list[dict[str, str]] = []
    for file_path in example_files:
        path = Path(file_path)
        content = path.read_text()
        loaded_files.append({"path": str(path), "content": content})
    return loaded_files


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
    *,
    example_files: Sequence[str | Path] | None = None,
    example_count: int = DEFAULT_EXAMPLE_COUNT,
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
    example_files : Sequence[str or Path] or None, default None
        Optional file paths used to ground the generated examples.
    example_count : int, default 3
        Number of examples to generate.
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
    """
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
        example_files=example_files,
        example_count=example_count,
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
    examples: Sequence[ExampleDataStructure],
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
    examples : Sequence[ExampleDataStructure]
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
        raise ValueError("At least one ExampleDataStructure instance is required.")
    if not openai_settings.default_model:
        raise ValueError("OpenAISettings.default_model is required for agent runs.")
    prompt_description = optimize_extractor_prompt_with_agent(
        openai_settings,
        prompt,
        extraction_classes,
        additional_context=additional_context,
    )
    request_text = _format_extractor_config_request_with_examples(
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
