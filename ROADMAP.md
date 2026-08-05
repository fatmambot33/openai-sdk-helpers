# Roadmap

Every roadmap item must support `PRODUCT.md` and preserve a small, typed, predictable public API.

## Completed — Codex plugin foundation

- [x] Define the minimal typed plugin protocol.
- [x] Add deterministic plugin and command registration.
- [x] Support package discovery through `openai_sdk_helpers.codex` entry points.
- [x] Add focused registry tests.
- [x] Export the stable Codex surface from the package root.
- [x] Add lifecycle hooks for startup and shutdown.
- [x] Add async command support without duplicating the synchronous API.
- [x] Add atomic rollback when plugin setup fails.
- [x] Add a runnable first-plugin example and packaging guide.

## Completed — production hardening

- [x] Define plugin compatibility and deprecation policy.
- [x] Add isolated discovery failure reporting.
- [x] Add structured plugin metadata and capability inspection.
- [x] Add CLI commands to list plugins and commands.
- [x] Test installed entry-point discovery end to end.
- [x] Add migration guidance before the next minor release.

## Next — official integrations

Official integrations remain optional and use the same plugin contract. They should be proposed only when a concrete user workflow justifies them:

- Responses
- Agents
- MCP
- File Search and Vector Stores
- Realtime
- Images and Audio

## Release gates

A milestone is complete only when:

1. Tests, type checks, docstring checks, and package builds pass.
2. Public APIs are documented with runnable examples.
3. Backward compatibility impact is explicit.
4. No OpenAI API call is hidden behind surprising defaults.
5. The core package remains usable without optional integrations.
