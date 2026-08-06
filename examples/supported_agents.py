"""Build an Agents SDK agent without making an API call."""

from __future__ import annotations

from openai_sdk_helpers.agent import AgentConfiguration, AgentRegistry


def main() -> None:
    """Run the deterministic Agents configuration workflow."""
    registry = AgentRegistry()
    configuration = AgentConfiguration(
        name="example_agent",
        instructions="Answer clearly and preserve the caller's intent.",
        model="example-model",
    )

    registry.register(configuration)
    helper = registry.get("example_agent").gen_agent()
    sdk_agent = helper.get_agent()

    assert helper.name == "example_agent"
    assert helper.instructions_text.startswith("Answer clearly")
    assert sdk_agent.name == "example_agent"
    assert sdk_agent.model == "example-model"
    print("Agents configuration workflow passed.")


if __name__ == "__main__":
    main()
