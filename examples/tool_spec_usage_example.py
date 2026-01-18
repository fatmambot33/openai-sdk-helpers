"""Example demonstrating ToolSpec usage patterns.

This example shows practical scenarios where ToolSpec and build_tool_definition_list
provide value over inline tool definitions:
1. Multi-tool response configurations
2. Reusable tool definitions across configurations
3. Dynamic tool composition
"""

from openai_sdk_helpers import (
    ResponseConfiguration,
    ToolSpec,
    build_tool_definition_list,
)
from openai_sdk_helpers.structure import (
    PromptStructure,
    SummaryStructure,
    ValidationResultStructure,
)

# Example 1: Multi-Tool Configuration
# =====================================


def example_multi_tool_config():
    """Demonstrate clear multi-tool configuration with ToolSpec."""
    print("\n=== Example 1: Multi-Tool Configuration ===")

    # Before: Inline definitions with repeated structure
    # tools = [
    #     PromptStructure.response_tool_definition(
    #         "web_agent", "Run a web research workflow"
    #     ),
    #     PromptStructure.response_tool_definition(
    #         "vector_agent", "Run a vector search workflow"
    #     ),
    #     SummaryStructure.response_tool_definition(
    #         "summarize", "Generate a summary"
    #     ),
    # ]

    # After: Named specifications with explicit metadata
    tool_specs = [
        ToolSpec(
            input_structure=PromptStructure,
            output_structure=PromptStructure,
            tool_name="web_agent",
            tool_description="Run a web research workflow for the provided prompt.",
        ),
        ToolSpec(
            input_structure=PromptStructure,
            output_structure=PromptStructure,
            tool_name="vector_agent",
            tool_description="Run a vector search workflow for the provided prompt.",
        ),
        ToolSpec(
            input_structure=SummaryStructure,
            output_structure=SummaryStructure,
            tool_name="summarize",
            tool_description="Generate a comprehensive summary with topic breakdown.",
        ),
    ]

    tools = build_tool_definition_list(tool_specs)

    configuration = ResponseConfiguration(
        name="multi_agent_assistant",
        instructions="You coordinate between multiple specialized agents.",
        tools=tools,
        input_structure=None,
        output_structure=None,
    )

    print(f"Created configuration '{configuration.name}' with {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")


# Example 2: Reusable Tool Definitions
# =====================================

# Define common tools once
RESEARCH_TOOLS = [
    ToolSpec(
        input_structure=PromptStructure,
        output_structure=PromptStructure,
        tool_name="web_agent",
        tool_description="Run a web research workflow for the provided prompt.",
    ),
    ToolSpec(
        input_structure=PromptStructure,
        output_structure=PromptStructure,
        tool_name="vector_agent",
        tool_description="Run a vector search workflow for the provided prompt.",
    ),
]

ANALYSIS_TOOLS = [
    ToolSpec(
        input_structure=SummaryStructure,
        output_structure=SummaryStructure,
        tool_name="summarize",
        tool_description="Generate a comprehensive summary with topic breakdown.",
    ),
    ToolSpec(
        input_structure=ValidationResultStructure,
        output_structure=ValidationResultStructure,
        tool_name="validate",
        tool_description="Validate the results and provide pass/fail status.",
    ),
]


def example_reusable_tools():
    """Demonstrate reusing tool definitions across multiple configurations."""
    print("\n=== Example 2: Reusable Tool Definitions ===")

    # Research-focused configuration
    research_config = ResponseConfiguration(
        name="research_assistant",
        instructions="You perform research using web and vector search.",
        tools=build_tool_definition_list(RESEARCH_TOOLS),
        input_structure=None,
        output_structure=None,
    )

    # Analysis-focused configuration
    analysis_config = ResponseConfiguration(
        name="analysis_assistant",
        instructions="You analyze and validate research results.",
        tools=build_tool_definition_list(ANALYSIS_TOOLS),
        input_structure=None,
        output_structure=None,
    )

    # Full pipeline configuration combining both
    full_config = ResponseConfiguration(
        name="full_pipeline_assistant",
        instructions="You handle the full research and analysis pipeline.",
        tools=build_tool_definition_list(RESEARCH_TOOLS + ANALYSIS_TOOLS),
        input_structure=None,
        output_structure=None,
    )

    print(f"Research configuration has {len(research_config.tools or [])} tools")
    print(f"Analysis configuration has {len(analysis_config.tools or [])} tools")
    print(f"Full pipeline configuration has {len(full_config.tools or [])} tools")


# Example 3: Dynamic Tool Composition
# ====================================


def example_dynamic_tools():
    """Demonstrate dynamic tool selection based on requirements."""
    print("\n=== Example 3: Dynamic Tool Composition ===")

    # Tool registry
    TOOL_REGISTRY = {
        "web": ToolSpec(
            input_structure=PromptStructure,
            output_structure=PromptStructure,
            tool_name="web_agent",
            tool_description="Run a web research workflow.",
        ),
        "vector": ToolSpec(
            input_structure=PromptStructure,
            output_structure=PromptStructure,
            tool_name="vector_agent",
            tool_description="Run a vector search workflow.",
        ),
        "summarize": ToolSpec(
            input_structure=SummaryStructure,
            output_structure=SummaryStructure,
            tool_name="summarize",
            tool_description="Generate summaries.",
        ),
        "validate": ToolSpec(
            input_structure=ValidationResultStructure,
            output_structure=ValidationResultStructure,
            tool_name="validate",
            tool_description="Validate results.",
        ),
    }

    # Function to build configuration with selected tools
    def build_config_with_tools(name: str, required_tools: list[str]):
        selected_specs = [TOOL_REGISTRY[tool_name] for tool_name in required_tools]
        return ResponseConfiguration(
            name=name,
            instructions=f"Assistant with {', '.join(required_tools)} capabilities.",
            tools=build_tool_definition_list(selected_specs),
            input_structure=None,
            output_structure=None,
        )

    # Create different configurations dynamically
    basic_config = build_config_with_tools("basic", ["web"])
    research_config = build_config_with_tools("research", ["web", "vector"])
    full_config = build_config_with_tools(
        "full", ["web", "vector", "summarize", "validate"]
    )

    print(f"Basic configuration: {len(basic_config.tools or [])} tool(s)")
    print(f"Research configuration: {len(research_config.tools or [])} tool(s)")
    print(f"Full configuration: {len(full_config.tools or [])} tool(s)")


# Example 4: Comparing Approaches
# ================================


def example_comparison():
    """Compare inline vs ToolSpec approach."""
    print("\n=== Example 4: Approach Comparison ===")

    # Inline approach (current)
    inline_tools = [
        PromptStructure.response_tool_definition(
            "web_agent", tool_description="Run a web research workflow"
        ),
        PromptStructure.response_tool_definition(
            "vector_agent", tool_description="Run a vector search workflow"
        ),
    ]

    # ToolSpec approach (new)
    tool_specs = [
        ToolSpec(
            input_structure=PromptStructure,
            output_structure=PromptStructure,
            tool_name="web_agent",
            tool_description="Run a web research workflow",
        ),
        ToolSpec(
            input_structure=PromptStructure,
            output_structure=PromptStructure,
            tool_name="vector_agent",
            tool_description="Run a vector search workflow",
        ),
    ]
    toolspec_tools = build_tool_definition_list(tool_specs)

    # Both produce the same result
    print("Inline approach: tools defined directly in list")
    print("ToolSpec approach: explicit named structures")
    print(f"\nBoth produce {len(inline_tools)} tool definitions")
    print(
        f"Tool 1 name matches: {inline_tools[0]['name'] == toolspec_tools[0]['name']}"
    )
    print(
        f"Tool 2 name matches: {inline_tools[1]['name'] == toolspec_tools[1]['name']}"
    )

    print("\nBenefits of ToolSpec approach:")
    print("  ✓ Named fields (no tuple ordering issues)")
    print("  ✓ Reusable tool specs")
    print("  ✓ Type-safe with StructureType")
    print("  ✓ Cleaner for multi-tool scenarios")
    print("  ✓ Better for dynamic composition")
    print("  ✓ Supports separate input/output structures")


# Example 5: Tools with Different Input/Output Structures
# ========================================================


def example_different_io_structures():
    """Demonstrate tools with different input and output structures."""
    print("\n=== Example 5: Different Input/Output Structures ===")

    # Some tools accept one type and return another
    tool_specs = [
        # Tool that accepts a prompt and returns a summary
        ToolSpec(
            input_structure=PromptStructure,
            tool_name="summarizer",
            tool_description="Generate a summary from the provided prompt",
            output_structure=SummaryStructure,
        ),
        # Tool that accepts a prompt and validates it
        ToolSpec(
            input_structure=PromptStructure,
            tool_name="validator",
            tool_description="Validate the provided prompt",
            output_structure=ValidationResultStructure,
        ),
        # Tool that accepts and returns the same type
        ToolSpec(
            input_structure=PromptStructure,
            tool_name="processor",
            tool_description="Process the prompt",
            output_structure=PromptStructure,
        ),
    ]

    tools = build_tool_definition_list(tool_specs)

    print(f"\nBuilt {len(tools)} tools with mixed I/O structures:")
    for i, (spec, tool) in enumerate(zip(tool_specs, tools), 1):
        input_structure = spec.input_structure.__name__
        output_structure = (
            spec.output_structure.__name__ if spec.output_structure else input_structure
        )
        print(f"\n{i}. {tool['name']}")
        print(f"   Input:  {input_structure}")
        print(f"   Output: {output_structure}")
        print(f"   {tool['description']}")

    print("\nNote: output_structure is for documentation/reference.")
    print("OpenAI tool definitions only use the input structure (parameters).")


# Main execution
# ==============


def main():
    """Run all examples."""
    print("=" * 70)
    print("ToolSpec Usage Examples")
    print("=" * 70)

    example_multi_tool_config()
    example_reusable_tools()
    example_dynamic_tools()
    example_comparison()
    example_different_io_structures()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
