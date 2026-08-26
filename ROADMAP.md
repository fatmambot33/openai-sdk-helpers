# Roadmap

Every roadmap item must support `PRODUCT.md` and preserve a small, typed,
predictable, SDK-first public API. The dependency-ordered execution tracker is
issue #147.

## 0.8.0 — platform foundation

Published and verified on PyPI.

- [x] #132 — make optional integrations truly optional.
- [x] #133 — migrate PyPI publishing to Trusted Publishing and protect the
      GitHub `pypi` environment.
- [x] #134 — replace duplicated feature documentation with a capability matrix.
- [x] #135 — execute supported examples and package smoke tests from the wheel.
- [x] #136 — define confidential vulnerability reporting and security policy.
- [x] #137 — approve, merge, publish, and verify 0.8.0.

The 0.8.0 release established:

- the completed Codex plugin protocol, lifecycle, discovery, metadata,
  inspection, CLI, compatibility policy, and migration guidance;
- minimal `core`, `extract`, `ui`, and `all` installation profiles;
- OIDC-only PyPI publication with build-only rehearsal, SBOM, and attestations;
- canonical capabilities, installation, security, and release documentation;
- Python 3.10–3.13, minimum/latest SDK, clean-install, and installed-wheel gates.

## 0.8.1 — shared runtime contracts

Published and verified on PyPI.

- [x] #138 — add a typed operation context and observability hooks.
- [x] #139 — define Responses continuation and Agents session semantics.

The runtime layer reuses official SDK tracing and session behavior rather than
building a parallel platform. Context is optional, diagnostics redact sensitive
content by default, ambiguous state ownership fails before SDK execution, and
local persistence remains explicitly caller-owned.

## 0.9.x — retrieval consolidation

0.9.0 was published through the protected PyPI Trusted Publishing path and
0.9.1 hardened the retrieval contract after delayed review findings.

- [x] #140 — approve one public retrieval API and migration map.
- [x] #141 — implement typed file and vector-store lifecycle helpers.
- [x] #142 — add File Search configuration and normalized results.

The retrieval surface now provides dependency-injected synchronous and
asynchronous Files and Vector Stores lifecycle operations, direct vector-store
search, typed common attribute filters, Responses and Agents File Search
adapters, normalized results and citations, and raw official SDK escape hatches.
Legacy retrieval imports remain available for compatibility and migration is
explicit rather than automatic.

## 0.10.0 — Realtime API helpers

- [ ] #145 — add thin typed server-side session configuration and lifecycle
      helpers over official SDK objects.
- [ ] #146 — add normalized events, explicit tool/cancellation helpers, and
      deterministic recorded-event/fake-session fixtures.

The scope excludes browser UI, audio-device management, a replacement transport,
a parallel event/state framework, and protocol reimplementation. Raw official
SDK sessions, clients, transports, and events remain accessible.

## Release gates

A milestone is complete only when:

1. tests, type checks, docstring checks, link checks, and package builds pass;
2. public APIs are typed, documented, and represented by runnable examples;
3. backward compatibility and migration impact are explicit;
4. network calls, destructive actions, ownership, and cleanup are explicit;
5. the core package remains usable without optional integrations;
6. built-wheel and supported-example smoke tests pass without credentials;
7. security-sensitive changes complete `docs/release-checklist.md` review;
8. publication uses protected OIDC identity and verified immutable artifacts.

Do not start a blocked phase, weaken a gate, or claim a release before external
owner controls and publication verification are complete. Do not add a helper
that primarily duplicates a general-purpose protocol or transport layer already
owned by the official SDK ecosystem.
