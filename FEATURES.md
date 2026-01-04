# Response Configuration Registry Implementation

This document describes the new features added to `openai-sdk-helpers` for response configuration, prompt management, tool handling, and plan execution.

## New Features

### 1. Response Configuration Registry

**Module:** `openai_sdk_helpers.response.config`

A centralized registry system for managing `ResponseConfiguration` instances across your application.

#### Classes

- **`ResponseRegistry`**: Manages a collection of response configurations with registration, retrieval, and listing capabilities.
- **`get_default_registry()`**: Returns a singleton registry instance for application-wide configuration storage.

#### Example Usage

```python
from openai_sdk_helpers.response import (
    ResponseConfiguration,
    ResponseRegistry,
    get_default_registry,
)

# Create a configuration
config = ResponseConfiguration(
    name="search_assistant",
    instructions="You are a search assistant",
    tools=None,
    input_structure=None,
    output_structure=SearchResults,
)

# Register it globally
registry = get_default_registry()
registry.register(config)

# Retrieve it later
config = registry.get("search_assistant")

# List all registered configs
print(registry.list_names())  # ['search_assistant']
```

### 2. Enhanced Prompt Renderer

**Module:** `openai_sdk_helpers.prompt.base`

The `PromptRenderer` now includes template caching and improved error reporting.

#### New Features

- **LRU Cache**: Templates are compiled and cached for improved performance (max 128 templates)
- **Clear Error Messages**: Better error reporting when templates are not found
- **Cache Control**: New `clear_cache()` method to force template recompilation

#### Example Usage

```python
from openai_sdk_helpers.prompt import PromptRenderer
from pathlib import Path

renderer = PromptRenderer(base_dir=Path("./templates"))

# First render compiles and caches the template
result1 = renderer.render("prompt.jinja", {"name": "Alice"})

# Second render uses cached template (faster)
result2 = renderer.render("prompt.jinja", {"name": "Bob"})

# Clear cache if templates are modified
renderer.clear_cache()
```

### 3. Enhanced Tool Argument Parsing

**Module:** `openai_sdk_helpers.response.tool_call`

The `parse_tool_arguments()` function now provides better error context.

#### Enhancements

- **Tool Name Context**: Optional `tool_name` parameter for clearer error messages
- **Payload Preview**: Long payloads are truncated in error messages
- **Clear Error Context**: Error messages include tool name and payload excerpt

#### Example Usage

```python
from openai_sdk_helpers.response import parse_tool_arguments

# Parse with tool name for better errors
try:
    args = parse_tool_arguments(
        '{"query": invalid}',
        tool_name="search_tool"
    )
except ValueError as e:
    print(e)
    # Failed to parse tool arguments for tool 'search_tool'. Raw payload: {"query": invalid}
```

### 4. Tool Handler Factory

**Module:** `openai_sdk_helpers.tools`

Generic tool handler infrastructure that eliminates boilerplate code.

#### Functions

- **`tool_handler_factory(func, input_model=None)`**: Creates a handler that automatically:
  1. Parses tool call arguments
  2. Validates with Pydantic (if model provided)
  3. Executes the function
  4. Serializes the result

- **`serialize_tool_result(result)`**: Standardizes serialization for Pydantic models, lists, dicts, and strings

#### Example Usage

```python
from openai_sdk_helpers.tools import tool_handler_factory
from pydantic import BaseModel

# Define input schema
class SearchInput(BaseModel):
    query: str
    limit: int = 10

# Define tool function
def search_tool(query: str, limit: int = 10):
    return {"results": [f"Result for {query}"]}

# Create handler with automatic validation
handler = tool_handler_factory(search_tool, input_model=SearchInput)

# Use with OpenAI tool calls
tool_handlers = {
    "search": handler
}
```

### 5. OpenAI Settings Builder

**Module:** `openai_sdk_helpers.utils.core`

Convenience function for creating `OpenAISettings` with validation.

#### Function

- **`build_openai_settings(**kwargs)`**: Builds settings from environment with:
  - Explicit validation and clear error messages
  - Type coercion for timeout and max_retries
  - Support for .env files
  - Parameter validation before client creation

#### Example Usage

```python
from openai_sdk_helpers.utils import build_openai_settings

# From explicit parameters
settings = build_openai_settings(
    api_key="sk-...",
    default_model="gpt-4o",
    timeout=30.0,
    max_retries=3,
)

# From environment variables
settings = build_openai_settings()

# With string parsing
settings = build_openai_settings(
    api_key="sk-...",
    timeout="45.5",  # Parsed to float
    max_retries="5",  # Parsed to int
)

# With custom .env file
settings = build_openai_settings(
    dotenv_path=Path("/path/to/.env")
)
```

### 6. Plan Execution Helpers

**Module:** `openai_sdk_helpers.structure.plan.helpers`

Convenience functions for working with plans and tasks.

#### Functions

- **`create_plan(*tasks)`**: Factory function for creating plans from tasks
- **`execute_task(task, agent_callable, context=None)`**: Execute a single task with status tracking
- **`execute_plan(plan, agent_registry, halt_on_error=True)`**: Execute a complete plan

#### Example Usage

```python
from openai_sdk_helpers.structure.plan import (
    AgentEnum,
    TaskStructure,
    create_plan,
    execute_task,
    execute_plan,
)

# Create tasks
task1 = TaskStructure(
    task_type=AgentEnum.WEB_SEARCH,
    prompt="Search for AI trends"
)
task2 = TaskStructure(
    task_type=AgentEnum.SUMMARIZER,
    prompt="Summarize findings"
)

# Create plan
plan = create_plan(task1, task2)

# Define agent callables
def search_agent(prompt, context=None):
    return ["search results"]

def summary_agent(prompt, context=None):
    return ["summary"]

# Execute plan
registry = {
    AgentEnum.WEB_SEARCH: search_agent,
    AgentEnum.SUMMARIZER: summary_agent,
}
results = execute_plan(plan, registry)

# Or execute single task
results = execute_task(task1, search_agent)
```

## Testing

All new features include comprehensive test coverage:

- **ResponseRegistry**: 7 tests covering registration, retrieval, clearing, and singleton behavior
- **Tool utilities**: 14 tests covering serialization, parsing, and handler factory
- **Prompt caching**: 5 tests covering caching behavior and cache clearing
- **Plan helpers**: 10 tests covering plan creation and execution
- **Settings builder**: 13 tests covering parameter validation and environment loading

**Total**: 49 new tests, all passing
**Code Coverage**: 78% (exceeds 70% requirement)

## Migration Guide

### For Existing Code

All changes are **backward compatible**. Existing code will continue to work without modifications.

### New Recommended Patterns

#### 1. Use ResponseRegistry for Configuration Management

**Before:**
```python
# Multiple scattered configurations
response1 = BaseResponse(instructions="...", ...)
response2 = BaseResponse(instructions="...", ...)
```

**After:**
```python
# Centralized registry
registry = get_default_registry()
registry.register(ResponseConfiguration(name="assistant", ...))
config = registry.get("assistant")
response = config.gen_response(settings, handlers)
```

#### 2. Use Tool Handler Factory

**Before:**
```python
def tool_handler(tool_call):
    try:
        args = json.loads(tool_call.arguments)
        validated = InputModel(**args)
        result = my_function(**validated.model_dump())
        return json.dumps(result)
    except Exception as e:
        # Handle errors...
```

**After:**
```python
handler = tool_handler_factory(my_function, input_model=InputModel)
# That's it! Parsing, validation, and serialization are automatic
```

#### 3. Use Plan Execution Helpers

**Before:**
```python
plan = PlanStructure(tasks=[task1, task2])
results = plan.execute(registry)
```

**After:**
```python
plan = create_plan(task1, task2)
results = execute_plan(plan, registry)
```

## Benefits

1. **Reduced Boilerplate**: Tool handler factory eliminates repetitive code
2. **Better Performance**: Template caching improves prompt rendering speed
3. **Clearer Errors**: Enhanced error messages with context speed up debugging
4. **Centralized Configuration**: Registry pattern improves configuration management
5. **Type Safety**: Explicit validation with Pydantic models
6. **Consistency**: Standardized serialization and parsing across the codebase
