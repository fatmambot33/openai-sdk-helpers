<div align="center">

# openai-sdk-helpers

[![PyPI version](https://img.shields.io/pypi/v/openai-sdk-helpers.svg)](https://pypi.org/project/openai-sdk-helpers/)
[![Python versions](https://img.shields.io/pypi/pyversions/openai-sdk-helpers.svg)](https://pypi.org/project/openai-sdk-helpers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Shared primitives for composing OpenAI workflows: high-level agent abstractions
(via `openai-agents` SDK) and low-level response handling (via `openai` SDK),
plus structures, prompt rendering, and reusable utilities.

[Installation](#installation) •
[Quickstart](#quickstart) •
[Features](#features) •
[Documentation](#key-modules) •
[Contributing](#contributing)

</div>

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quickstart](#quickstart)
  - [Text utilities](#text-utilities)
  - [Centralized OpenAI configuration](#centralized-openai-configuration)
- [Advanced Usage](#advanced-usage)
- [Development](#development)
- [Project Structure](#project-structure)
- [Key modules](#key-modules)
- [Contributing](#contributing)
- [License](#license)
- [Troubleshooting](#troubleshooting)

## Overview

`openai-sdk-helpers` packages the common building blocks required to assemble agent-driven
applications. The library intentionally focuses on reusable primitives—data
structures, configuration helpers, and orchestration utilities—while leaving
application-specific prompts and tools to the consuming project.

**Important**: This library integrates with **two distinct OpenAI SDKs**:
- **`openai-agents`** - Used by the `agent` module for high-level agent workflows with automatic tool handling and streaming
- **`openai`** - Used by the `response` module for direct API interactions with fine-grained control over responses

The `agent` module provides a higher-level abstraction for building agents, while the `response` module offers lower-level control for custom response handling workflows.

### Key Features

#### Agent Module (Built on `openai-agents` SDK)
- **Agent wrappers** with synchronous and asynchronous entry points
- **Prompt rendering** powered by Jinja2 for dynamic agent instructions
- **Vector and web search flows** that coordinate planning, execution, and
  reporting with built-in concurrency control
- **Reusable text agents** for common tasks:
  - **SummarizerAgent**: Generate concise summaries from provided text
  - **TranslatorAgent**: Translate text into target languages
  - **ValidatorAgent**: Check inputs and outputs against safety guardrails

#### Response Module (Built on `openai` SDK)
- **Response handling utilities** for direct API control with fine-grained message management
- **Tool execution framework** with custom handlers and structured outputs
- **Session persistence** for saving and restoring conversation history

#### Shared Components
- **Typed structures** using Pydantic for prompts, responses, and search workflows 
  to ensure predictable inputs and outputs
- **OpenAI configuration management** with environment variable and `.env` file support
- **Vector storage abstraction** for seamless integration with OpenAI vector stores
- **Type-safe interfaces** with full type hints and `py.typed` marker for external projects

## Requirements

- Python 3.10 or higher
- OpenAI API key (set via `OPENAI_API_KEY` environment variable)

**Note**: This package depends on both:
- `openai` - The standard OpenAI Python SDK
- `openai-agents` - The OpenAI Agents SDK for high-level agent workflows

Both are automatically installed when you install `openai-sdk-helpers`.

## Installation

Install the package directly from PyPI:

```bash
pip install openai-sdk-helpers
```

The package ships with type information via `py.typed`, enabling full type checking
support in your IDE and with tools like mypy and pyright.

### Development Installation

For local development with editable sources and optional dev dependencies:

```bash
# Clone the repository
git clone https://github.com/fatmambot33/openai-sdk-helpers.git
cd openai-sdk-helpers

# Install in editable mode with dev dependencies
pip install -e .[dev]
```

The dev dependencies include:
- `pydocstyle` - Docstring style checking
- `pyright` - Static type checking
- `black` - Code formatting
- `pytest` and `pytest-cov` - Testing and coverage

## Quickstart

### Basic Vector Search

Create a vector search workflow by wiring your own prompt templates and
preferred model configuration. This example uses the `agent` module built on `openai-agents` SDK:

```python
from pathlib import Path
from openai_sdk_helpers.agent.search.vector import VectorSearch

# Point to your custom prompt directory
prompts = Path("./prompts")

# Create and configure the vector search agent
vector_search = VectorSearch(
    prompt_dir=prompts,
    default_model="gpt-4o-mini"
)

# Run a synchronous search query
report = vector_search.run_agent_sync("Explain quantum entanglement for beginners")
print(report.report)
```

**Note**: The vector search workflow requires prompt templates for each agent
(`vector_planner.jinja`, `vector_search.jinja`, and `vector_writer.jinja`). 
If your `prompt_dir` doesn't contain these files, agent construction will fail 
with a `FileNotFoundError`.

### Text utilities

The built-in text helpers provide lightweight single-step agents for common tasks.
These use the `agent` module built on `openai-agents` SDK:

```python
from openai_sdk_helpers.agent import (
    SummarizerAgent,
    TranslatorAgent,
    ValidatorAgent,
)

# Initialize agents with a default model
summarizer = SummarizerAgent(default_model="gpt-4o-mini")
translator = TranslatorAgent(default_model="gpt-4o-mini")
validator = ValidatorAgent(default_model="gpt-4o-mini")

# Generate a summary
summary = summarizer.run_sync("Long-form content to condense...")
print(summary.text)

# Translate text
translation = translator.run_sync("Bonjour", target_language="English")
print(translation)

# Validate against guardrails
validation = validator.run_sync(
    "Share meeting notes with names removed",
    agent_output=summary.text
)
print(validation.is_safe)
```

**Important**: These text helpers ship with default prompt templates under
`src/openai_sdk_helpers/prompt`, so you do **not** need to create placeholder
files when installing from PyPI. Only pass a `prompt_dir` when you have custom
templates you want to use instead.

### Centralized OpenAI configuration

`OpenAISettings` provides a centralized way to manage OpenAI SDK configuration
across your application:

```python
from openai_sdk_helpers import OpenAISettings

# Load from environment variables or a local .env file
settings = OpenAISettings.from_env()

# Create an OpenAI client with loaded settings
client = settings.create_client()

# Reuse the default model across agents
from openai_sdk_helpers.agent.search.vector import VectorSearch

vector_search = VectorSearch(
    prompt_dir=prompts,
    default_model=settings.default_model or "gpt-4o-mini"
)
```

**Supported Environment Variables:**
- `OPENAI_API_KEY` (required) - Your OpenAI API key
- `OPENAI_ORG_ID` - Organization identifier
- `OPENAI_PROJECT_ID` - Project identifier for billing
- `OPENAI_BASE_URL` - Custom base URL for self-hosted deployments
- `OPENAI_MODEL` - Default model name
- `OPENAI_TIMEOUT` - Request timeout in seconds
- `OPENAI_MAX_RETRIES` - Maximum retry attempts

Pass uncommon OpenAI client keyword arguments (such as `default_headers`,
`http_client`, or custom `base_url` proxies) through `extra_client_kwargs`
when instantiating `OpenAISettings`.

### Direct Response Control (Response Module)

For more fine-grained control over API interactions, use the `response` module built on the standard `openai` SDK. This gives you direct access to message history, tool handlers, and custom response parsing:

```python
from openai_sdk_helpers.response import BaseResponse
from openai_sdk_helpers import OpenAISettings

# Configure OpenAI settings
settings = OpenAISettings.from_env()

# Create a response handler with custom instructions
response = BaseResponse(
    instructions="You are a helpful code review assistant.",
    tools=None,  # Or provide custom tool definitions
    output_structure=None,  # Or a Pydantic model for structured output
    tool_handlers={},  # Map tool names to handler functions
    openai_settings=settings
)

# Execute and get a response
result = response.run_sync("Review this Python code for best practices...")
print(result)

# Clean up
response.close()
```

**Key Differences:**
- **Agent Module**: Higher-level abstraction with built-in streaming, automatic tool handling, and agent-specific workflows
- **Response Module**: Lower-level control with manual message management, custom tool handlers, and direct API access

## Advanced Usage

### Custom Prompt Templates

Create custom Jinja2 templates for specialized agent behaviors:

```python
from pathlib import Path
from openai_sdk_helpers.agent import SummarizerAgent

# Use custom prompt templates
custom_prompts = Path("./my_prompts")
summarizer = SummarizerAgent(
    prompt_dir=custom_prompts,
    default_model="gpt-4o"
)
```

### Asynchronous Execution

All agents support both synchronous and asynchronous execution:

```python
import asyncio
from openai_sdk_helpers.agent import SummarizerAgent

async def main():
    summarizer = SummarizerAgent(default_model="gpt-4o-mini")
    
    # Run asynchronously
    result = await summarizer.run_agent(
        text="Long document to summarize...",
        metadata={"source": "example.pdf"}
    )
    print(result.text)

asyncio.run(main())
```

### Vector Storage Integration

Integrate with OpenAI vector stores for document search:

```python
from openai_sdk_helpers.vector_storage import VectorStorage
from openai_sdk_helpers.agent.search.vector import VectorSearch

# Create or connect to a vector store
storage = VectorStorage(store_name="my_documents")

# Use it with vector search
vector_search = VectorSearch(
    default_model="gpt-4o-mini",
    vector_storage=storage
)
```

## Development

The repository follows standard Python development practices with comprehensive
quality checks.

### Setting Up Your Development Environment

```bash
# Clone and install with dev dependencies
git clone https://github.com/fatmambot33/openai-sdk-helpers.git
cd openai-sdk-helpers
pip install -e .[dev]
```

### Running Quality Checks

Before opening a pull request, ensure all quality checks pass locally:

```bash
# Style and docstring checks
pydocstyle src

# Code formatting check
black --check .

# Apply formatting (if needed)
black .

# Static type checking
pyright src

# Unit tests with coverage (minimum 70% required)
pytest -q --cov=src --cov-report=term-missing --cov-fail-under=70
```

All checks must pass for changes to be merged.

## Project Structure

```
src/openai_sdk_helpers/
├── agent/              # Agent factories, orchestration, and search workflows
│   ├── base.py        # Base agent class with sync/async execution
│   ├── summarizer.py  # Text summarization agent
│   ├── translator.py  # Translation agent
│   ├── validation.py  # Input/output validation agent
│   ├── vector_search.py  # Multi-agent vector search workflow
│   └── coordinator_agent.py  # Coordinated multi-step workflows
├── prompt/             # Jinja2 template rendering utilities
├── response/           # Response parsing and transformation helpers
├── structure/          # Pydantic-based typed data structures
│   ├── base.py        # Base structure with schema generation
│   ├── plan/          # Task and plan structures
│   ├── summary.py     # Summary output structures
│   └── validation.py  # Validation result structures
├── vector_storage/     # Vector store abstraction layer
├── config.py          # OpenAI settings and configuration
└── utils/             # JSON serialization, logging, and helpers

tests/                  # Comprehensive unit test suite
```

## Key Modules

The package is organized around cohesive, reusable building blocks:

### Agent Modules (Built on `openai-agents` SDK)

These modules use the `openai-agents` SDK for high-level agent workflows with automatic streaming, tool handling, and conversation management.

- **`openai_sdk_helpers.agent.base.AgentBase`**  
  Base class for all agents with synchronous and asynchronous execution support.
  Handles prompt rendering, model configuration, and tool integration.

- **`openai_sdk_helpers.agent.search.vector.VectorSearch`**  
  Complete vector search workflow that coordinates planning, searching, and 
  reporting. Bundles `VectorSearchPlanner`, `VectorSearchTool`, and 
  `VectorSearchWriter` into a single entry point.

- **`openai_sdk_helpers.agent.coordinator_agent.ProjectManager`**  
  Orchestrates multi-step workflows with prompt creation, plan building, task 
  execution, and summarization. Persists intermediate artifacts to disk.

- **`openai_sdk_helpers.agent.summarizer.SummarizerAgent`**  
  Generates concise summaries from provided text using structured output.

- **`openai_sdk_helpers.agent.translator.TranslatorAgent`**  
  Translates text into target languages with optional context.

- **`openai_sdk_helpers.agent.validation.ValidatorAgent`**  
  Validates user inputs and agent outputs against safety guardrails.

### Response Module (Built on `openai` SDK)

These modules use the standard `openai` SDK for direct API interactions with fine-grained control over request/response cycles.

- **`openai_sdk_helpers.response.base.BaseResponse`**  
  Manages complete OpenAI API interaction lifecycle including input construction,
  tool execution, message history, and structured output parsing. Uses the 
  `client.responses.create()` API (from the OpenAI Responses API, distinct from 
  the Chat Completions API) for direct control over conversation flow.

- **`openai_sdk_helpers.response.runner`**  
  Convenience functions for executing response workflows with automatic cleanup
  in both synchronous and asynchronous contexts.

### Configuration and Data Structures (Shared)

- **`openai_sdk_helpers.config.OpenAISettings`**  
  Centralizes OpenAI API configuration with environment variable support.
  Creates configured OpenAI clients with consistent settings.

- **`openai_sdk_helpers.structure.BaseStructure`**  
  Pydantic-based foundation for all structured outputs. Provides JSON schema
  generation, validation, and serialization helpers.

- **`openai_sdk_helpers.structure.plan`**  
  Task and plan structures for multi-step workflows with status tracking.

### Utilities (Shared)

- **`openai_sdk_helpers.prompt.PromptRenderer`**  
  Jinja2-based template rendering for dynamic prompt generation.

- **`openai_sdk_helpers.vector_storage.VectorStorage`**  
  Minimal abstraction over OpenAI vector stores with search and file management.

- **`openai_sdk_helpers.utils`**  
  JSON serialization helpers, logging utilities, and common validation functions.

## Contributing

Contributions are welcome! We appreciate functional changes accompanied by
relevant tests and documentation.

### Guidelines

1. **Fork and clone** the repository
2. **Create a feature branch** from `main`
3. **Make your changes** following the project conventions
4. **Add or update tests** to cover your changes
5. **Run all quality checks** (see [Development](#development))
6. **Submit a pull request** with a clear description

### Code Style

- Follow **PEP 8** for Python code formatting
- Use **NumPy-style docstrings** as outlined in `AGENTS.md`
- Write clear, descriptive commit messages in the imperative mood
- Keep implementations Pythonic and maintainable

### Documentation

- Document all public classes, methods, and functions
- Include a `Methods` section in class docstrings listing public methods
- Add type hints to all function signatures
- Provide examples in docstrings where helpful

### Testing

- Write unit tests for new functionality
- Maintain minimum 70% code coverage
- Ensure all tests pass before submitting

See `AGENTS.md` for detailed contributing guidelines and conventions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## Troubleshooting

### Common Issues

**"OPENAI_API_KEY is required" error**

Ensure your OpenAI API key is set in the environment:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file in your project root:
```
OPENAI_API_KEY=your-api-key-here
```

**"Prompt template not found" error**

Vector search workflows require custom prompt templates. Either:
1. Create the required `.jinja` files in your `prompt_dir`
2. Omit the `prompt_dir` parameter to use built-in defaults (for text agents only)

**Import errors after installation**

Ensure you're using Python 3.10 or higher:
```bash
python --version
```

If using an older version, upgrade Python or create a virtual environment with
Python 3.10+.

**Type checking issues in your IDE**

The package ships with `py.typed` for full type support. If your IDE isn't
recognizing types:
1. Ensure your IDE's Python plugin is up to date
2. Restart your IDE after installing the package
3. Check that your IDE is using the correct Python interpreter

### Getting Help

- Check the [Key Modules](#key-modules) section for API documentation
- Review examples in the [Quickstart](#quickstart) and [Advanced Usage](#advanced-usage) sections
- Open an issue on GitHub for bugs or feature requests

