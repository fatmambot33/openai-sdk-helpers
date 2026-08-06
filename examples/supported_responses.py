"""Build and inspect a Responses configuration without an API call."""

from __future__ import annotations

from openai_sdk_helpers.response import ResponseConfiguration, ResponseRegistry


def main() -> None:
    """Run the deterministic Responses configuration workflow."""
    registry = ResponseRegistry()
    configuration = ResponseConfiguration(
        name="example_responder",
        instructions="Return a concise answer.",
        tools=[{"type": "web_search"}],
        input_structure=None,
        output_structure=None,
    )

    registry.register(configuration)
    registered = registry.get("example_responder")

    assert registered is configuration
    assert registered.name == "example_responder"
    assert registered.tools == [{"type": "web_search"}]
    print("Responses configuration workflow passed.")


if __name__ == "__main__":
    main()
