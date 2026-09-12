"""Validate managed Agents API construction without making an API call."""

from openai import OpenAI

from openai_sdk_helpers.managed_agents import (
    ManagedAgentsClient,
    managed_agents_available,
)


def main() -> None:
    """Construct the managed Agents facade against the installed SDK."""
    client = OpenAI(api_key="example-non-secret-key")
    assert managed_agents_available(client)

    helper = ManagedAgentsClient(client)
    assert helper.sdk_client is client
    assert helper.agents is client.beta.agents
    assert helper.sessions is client.beta.agents.sessions


if __name__ == "__main__":
    main()
