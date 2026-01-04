"""Example demonstrating all new response configuration registry features.

This example shows how to use:
1. ResponseRegistry for centralized configuration
2. Tool handler factory for automatic tool handling
3. Enhanced prompt rendering with caching
4. Plan execution helpers
5. Settings builder for configuration
"""

from pathlib import Path
from pydantic import BaseModel

from openai_sdk_helpers import (
    # Registry and configuration
    ResponseRegistry,
    ResponseConfiguration,
    get_default_registry,
    # Tool utilities
    tool_handler_factory,
    serialize_tool_result,
    # Settings and utilities
    build_openai_settings,
    # Plan execution
    create_plan,
    execute_task,
    execute_plan,
    TaskStructure,
    AgentEnum,
    # Prompt rendering
    PromptRenderer,
)


# Example 1: Response Registry
# ============================

def example_response_registry():
    """Demonstrate centralized response configuration management."""
    print("\n=== Example 1: Response Registry ===")
    
    # Get singleton registry
    registry = get_default_registry()
    
    # Create configurations
    search_config = ResponseConfiguration(
        name="search_assistant",
        instructions="You are a helpful search assistant",
        tools=None,
        input_structure=None,
        output_structure=None,
    )
    
    summary_config = ResponseConfiguration(
        name="summary_assistant",
        instructions="You create concise summaries",
        tools=None,
        input_structure=None,
        output_structure=None,
    )
    
    # Register configurations
    registry.register(search_config)
    registry.register(summary_config)
    
    # List all configurations
    print(f"Registered configs: {registry.list_names()}")
    
    # Retrieve a configuration
    config = registry.get("search_assistant")
    print(f"Retrieved config: {config.name}")
    
    # Clear for cleanup
    registry.clear()
    print("Registry cleared")


# Example 2: Tool Handler Factory
# ===============================

class SearchInput(BaseModel):
    """Validated input for search tool."""
    query: str
    limit: int = 10


class SearchOutput(BaseModel):
    """Structured output from search tool."""
    results: list[str]
    count: int


def search_function(query: str, limit: int = 10) -> SearchOutput:
    """Actual search implementation."""
    return SearchOutput(
        results=[f"Result {i} for {query}" for i in range(limit)],
        count=limit,
    )


def example_tool_handler_factory():
    """Demonstrate automatic tool handling with validation."""
    print("\n=== Example 2: Tool Handler Factory ===")
    
    # Create handler with automatic parsing, validation, and serialization
    handler = tool_handler_factory(search_function, input_model=SearchInput)
    
    # Simulate a tool call
    class MockToolCall:
        def __init__(self):
            self.arguments = '{"query": "AI trends", "limit": 3}'
            self.name = "search"
    
    tool_call = MockToolCall()
    result = handler(tool_call)
    print(f"Tool result (JSON): {result}")
    
    # Direct serialization example
    output = SearchOutput(results=["A", "B"], count=2)
    serialized = serialize_tool_result(output)
    print(f"Serialized output: {serialized}")


# Example 3: Prompt Rendering with Caching
# ========================================

def example_prompt_caching(tmp_path):
    """Demonstrate template caching for performance."""
    print("\n=== Example 3: Prompt Rendering with Caching ===")
    
    # Create template directory
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    template_file = template_dir / "greeting.jinja"
    template_file.write_text("Hello {{ name }}, welcome to {{ app }}!")
    
    # Create renderer with caching
    renderer = PromptRenderer(base_dir=template_dir)
    
    # First render - compiles and caches
    result1 = renderer.render("greeting.jinja", {"name": "Alice", "app": "MyApp"})
    print(f"First render: {result1}")
    
    # Second render - uses cache (faster)
    result2 = renderer.render("greeting.jinja", {"name": "Bob", "app": "MyApp"})
    print(f"Second render (cached): {result2}")
    
    # Clear cache if needed
    renderer.clear_cache()
    print("Cache cleared")


# Example 4: Settings Builder
# ===========================

def example_settings_builder():
    """Demonstrate simplified OpenAI settings configuration."""
    print("\n=== Example 4: Settings Builder ===")
    
    try:
        # Build settings with validation
        settings = build_openai_settings(
            api_key="sk-test-key-example",
            default_model="gpt-4o",
            timeout="30.5",  # String parsed to float
            max_retries="3",  # String parsed to int
        )
        
        print(f"API Key: {settings.api_key[:10]}...")
        print(f"Model: {settings.default_model}")
        print(f"Timeout: {settings.timeout} (type: {type(settings.timeout).__name__})")
        print(f"Max Retries: {settings.max_retries} (type: {type(settings.max_retries).__name__})")
    except ValueError as e:
        print(f"Configuration error: {e}")


# Example 5: Plan Execution Helpers
# =================================

def example_plan_execution():
    """Demonstrate convenient plan creation and execution."""
    print("\n=== Example 5: Plan Execution Helpers ===")
    
    # Create tasks
    task1 = TaskStructure(
        task_type=AgentEnum.WEB_SEARCH,
        prompt="Search for Python best practices"
    )
    task2 = TaskStructure(
        task_type=AgentEnum.SUMMARIZER,
        prompt="Summarize the search results"
    )
    
    # Create plan using helper
    plan = create_plan(task1, task2)
    print(f"Created plan with {len(plan)} tasks")
    
    # Define mock agent functions
    def search_agent(prompt, context=None):
        """Mock search agent."""
        return [f"Search results for: {prompt}"]
    
    def summary_agent(prompt, context=None):
        """Mock summary agent."""
        context_str = f" (with context: {context})" if context else ""
        return [f"Summary of: {prompt}{context_str}"]
    
    # Create agent registry
    registry = {
        AgentEnum.WEB_SEARCH: search_agent,
        AgentEnum.SUMMARIZER: summary_agent,
    }
    
    # Execute plan
    results = execute_plan(plan, registry)
    print(f"Plan results: {results}")
    
    # Execute single task
    single_task = TaskStructure(
        task_type=AgentEnum.WEB_SEARCH,
        prompt="Quick search"
    )
    task_results = execute_task(single_task, search_agent)
    print(f"Single task results: {task_results}")


# Main execution
# =============

def main():
    """Run all examples."""
    import tempfile
    
    print("=" * 60)
    print("Response Configuration Registry Features Demo")
    print("=" * 60)
    
    # Run examples
    example_response_registry()
    example_tool_handler_factory()
    
    # Use temporary directory for prompt caching example
    with tempfile.TemporaryDirectory() as tmp_dir:
        example_prompt_caching(Path(tmp_dir))
    
    example_settings_builder()
    example_plan_execution()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
