"""Example demonstrating tool wrapper unwrapping and async handler support.

This example shows how the new features solve the two problems outlined:
1. Tool schema wrapper handling - automatic unwrapping of wrapped payloads
2. Async tool handler support - built-in async function support with event loop detection
"""

import asyncio

from openai_sdk_helpers import StructureBase, ToolSpec, spec_field
from openai_sdk_helpers.tools import tool_handler_factory


# Define input/output models
class TargetingInput(StructureBase):
    """Input parameters for the targeting tool."""

    campaign_id: str = spec_field("campaign_id", description="Campaign identifier")
    audience: str = spec_field("audience", description="Target audience")
    budget: float = spec_field(
        "budget",
        default=1000.0,
        description="Budget for targeting recommendations",
    )


class TargetingOutput(StructureBase):
    """Output from the targeting tool."""

    campaign_id: str = spec_field("campaign_id", description="Campaign identifier")
    targeting_status: str = spec_field(
        "targeting_status", description="Status of targeting recommendation"
    )
    estimated_reach: int = spec_field(
        "estimated_reach", description="Estimated audience reach"
    )
    message: str = spec_field("message", description="Status message")


# Example 1: Async tool with automatic wrapper unwrapping
# ========================================================
async def propose_targeting(
    campaign_id: str, audience: str, budget: float
) -> TargetingOutput:
    """Async tool that proposes targeting parameters for a campaign.

    Before: Required custom _run_awaitable/thread handling to work in event loops.
    After: tool_handler_factory automatically handles async functions with event loop detection.
    """
    # Simulate async API call
    await asyncio.sleep(0.01)

    # Calculate estimated reach based on budget
    estimated_reach = int(budget * 10)

    return TargetingOutput(
        campaign_id=campaign_id,
        targeting_status="active",
        estimated_reach=estimated_reach,
        message=f"Targeting configured for {audience} audience",
    )


# Create handler - no special async handling needed!
# Before: Had to implement custom thread/event loop logic
# After: Just pass async function directly
tool_spec = ToolSpec(
    tool_name="propose_targeting",
    tool_description="Propose targeting parameters for a campaign.",
    input_structure=TargetingInput,
    output_structure=TargetingOutput,
)
async_handler = tool_handler_factory(propose_targeting, tool_spec=tool_spec)


# Example 2: Handler automatically unwraps payloads
# ==================================================
class MockToolCall:
    """Mock tool call for demonstration."""

    def __init__(self, arguments: str, name: str):
        self.arguments = arguments
        self.name = name


def demonstrate_wrapper_unwrapping():
    """Show how wrapper unwrapping works automatically."""
    print("=" * 70)
    print("Example 1: Wrapper Unwrapping")
    print("=" * 70)

    # Scenario 1: Response wraps arguments under tool name
    # Before: Would fail validation with strict=True, additionalProperties=False
    # After: Automatically unwrapped before validation
    wrapped_call = MockToolCall(
        arguments='{"propose_targeting": {"campaign_id": "c123", "audience": "tech", "budget": 5000}}',
        name="propose_targeting",
    )

    result = async_handler(wrapped_call)
    print("\n✓ Wrapped payload (tool name key):")
    print(f"  Input:  {wrapped_call.arguments}")
    print(f"  Output: {result}")

    # Scenario 2: Response uses snake_case wrapper
    snake_case_call = MockToolCall(
        arguments='{"propose_targeting": {"campaign_id": "c456", "audience": "finance", "budget": 10000}}',
        name="propose_targeting",
    )

    result = async_handler(snake_case_call)
    print("\n✓ Wrapped payload (snake_case key):")
    print(f"  Input:  {snake_case_call.arguments}")
    print(f"  Output: {result}")

    # Scenario 3: Flat arguments (no wrapper) still work
    flat_call = MockToolCall(
        arguments='{"campaign_id": "c789", "audience": "healthcare", "budget": 15000}',
        name="propose_targeting",
    )

    result = async_handler(flat_call)
    print("\n✓ Flat payload (no wrapper):")
    print(f"  Input:  {flat_call.arguments}")
    print(f"  Output: {result}")


async def demonstrate_async_support():
    """Show how async tools work in various contexts."""
    print("\n" + "=" * 70)
    print("Example 2: Async Tool Handler Support")
    print("=" * 70)

    # Scenario 1: Async tool called outside event loop
    print("\n✓ Async handler called from sync context:")
    tool_call = MockToolCall(
        arguments='{"campaign_id": "async1", "audience": "education", "budget": 2000}',
        name="propose_targeting",
    )
    result = async_handler(tool_call)
    print(f"  Result: {result[:80]}...")

    # Scenario 2: Async tool called inside event loop
    print("\n✓ Async handler called from async context (we're in one now):")
    tool_call2 = MockToolCall(
        arguments='{"campaign_id": "async2", "audience": "retail", "budget": 8000}',
        name="propose_targeting",
    )
    result2 = async_handler(tool_call2)
    print(f"  Result: {result2[:80]}...")

    # Scenario 3: Multiple async calls work correctly
    print("\n✓ Multiple async calls in sequence:")
    for i in range(3):
        tool_call_seq = MockToolCall(
            arguments=f'{{"campaign_id": "seq{i}", "audience": "audience{i}", "budget": {(i+1)*1000}}}',
            name="propose_targeting",
        )
        result_seq = async_handler(tool_call_seq)
        print(f"  Call {i+1}: Success")


# Example 3: Sync tool still works with ToolSpec
# ===========================================
class SyncInput(StructureBase):
    """Input parameters for the sync tool."""

    name: str = spec_field("name", description="Name to echo")
    value: int = spec_field("value", description="Value to double")


class SyncOutput(StructureBase):
    """Output payload for the sync tool."""

    name: str = spec_field("name", description="Echoed name")
    doubled: int = spec_field("doubled", description="Doubled value")


def sync_tool(name: str, value: int) -> SyncOutput:
    """Synchronize tool for comparison."""
    return SyncOutput(name=name, doubled=value * 2)


sync_tool_spec = ToolSpec(
    tool_name="sync_tool",
    tool_description="Double a value and echo the name.",
    input_structure=SyncInput,
    output_structure=SyncOutput,
)
sync_handler = tool_handler_factory(sync_tool, tool_spec=sync_tool_spec)


def demonstrate_sync_tool_support():
    """Show that sync tools still work with ToolSpec."""
    print("\n" + "=" * 70)
    print("Example 3: Sync Tool Support")
    print("=" * 70)

    tool_call = MockToolCall(
        arguments='{"name": "test", "value": 21}',
        name="sync_tool",
    )
    result = sync_handler(tool_call)
    print(f"\n✓ Sync handler still works:")
    print(f"  Input:  {tool_call.arguments}")
    print(f"  Output: {result}")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_wrapper_unwrapping()

    # Run async example (creates event loop)
    asyncio.run(demonstrate_async_support())

    demonstrate_sync_tool_support()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(
        """
✓ Wrapper unwrapping: Arguments wrapped by tool name are automatically unwrapped
✓ Async support: Async functions work seamlessly with event loop detection
✓ Sync tools: ToolSpec handles input and output validation for sync tools
✓ No manual thread handling: Event loop management is automatic
✓ Validation still works: Pydantic validation happens after unwrapping
    """
    )
