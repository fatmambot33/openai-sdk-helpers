"""LangExtract-powered document extraction helpers."""

from __future__ import annotations

from .._optional import import_optional_module

_extractor_module = import_optional_module(
    "openai_sdk_helpers.extract.extractor",
    dependency="langextract",
    extra="extract",
    feature="Document extraction",
)
_generator_module = import_optional_module(
    "openai_sdk_helpers.extract.generator",
    dependency="langextract",
    extra="extract",
    feature="Document extraction",
)

DocumentExtractor = _extractor_module.DocumentExtractor
EXTRACTOR_CONFIG_GENERATOR = _generator_module.EXTRACTOR_CONFIG_GENERATOR
EXTRACTOR_CONFIG_AGENT_INSTRUCTIONS = (
    _generator_module.EXTRACTOR_CONFIG_AGENT_INSTRUCTIONS
)
PROMPT_OPTIMIZER_AGENT_INSTRUCTIONS = (
    _generator_module.PROMPT_OPTIMIZER_AGENT_INSTRUCTIONS
)
generate_document_extractor_config = (
    _generator_module.generate_document_extractor_config
)
generate_document_extractor_config_with_agent = (
    _generator_module.generate_document_extractor_config_with_agent
)
optimize_extractor_prompt = _generator_module.optimize_extractor_prompt
optimize_extractor_prompt_with_agent = (
    _generator_module.optimize_extractor_prompt_with_agent
)

__all__ = [
    "DocumentExtractor",
    "EXTRACTOR_CONFIG_GENERATOR",
    "EXTRACTOR_CONFIG_AGENT_INSTRUCTIONS",
    "PROMPT_OPTIMIZER_AGENT_INSTRUCTIONS",
    "generate_document_extractor_config",
    "generate_document_extractor_config_with_agent",
    "optimize_extractor_prompt",
    "optimize_extractor_prompt_with_agent",
]
