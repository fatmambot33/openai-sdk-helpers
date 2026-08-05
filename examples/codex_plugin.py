"""Minimal first-class Codex plugin example."""

from __future__ import annotations

import asyncio

from openai_sdk_helpers import CodexPluginContext, CodexPluginRegistry


class GreetingPlugin:
    """Register synchronous and asynchronous greeting commands."""

    name = "greeting"

    def setup(self, context: CodexPluginContext) -> None:
        """Register commands exposed by this plugin."""

        context.add_command("greet", lambda name: f"Hello, {name}!")

        async def greet_async(name: str) -> str:
            return f"Hello asynchronously, {name}!"

        context.add_command("greet_async", greet_async)

    async def startup(self) -> None:
        """Prepare optional plugin resources."""

    async def shutdown(self) -> None:
        """Release optional plugin resources."""


async def main() -> None:
    """Run the example plugin."""

    registry = CodexPluginRegistry()
    registry.register(GreetingPlugin())

    async with registry:
        print(registry.run("greet", "Codex"))
        print(await registry.run_async("greet_async", "Codex"))


if __name__ == "__main__":
    asyncio.run(main())
