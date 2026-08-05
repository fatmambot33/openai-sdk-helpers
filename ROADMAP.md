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

## Next — production hardening

- [ ] Define plugin compatibility and deprecation policy.
- [ ] Add isolated discovery failure reporting.
- [ ] Add structured plugin metadata and capability inspection.
- [ ] Add CLI commands to list plugins and commands.
- [ ] Test installed entry-point discovery end to end.
- [ ] Add migration guidance before the next minor release.

## Later — official integrations

Official integrations should remain optional and use the same plugin contract:

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
