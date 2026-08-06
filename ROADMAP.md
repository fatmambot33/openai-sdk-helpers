# Roadmap

Every roadmap item must support `PRODUCT.md` and preserve a small, typed,
predictable, SDK-first public API. The dependency-ordered execution tracker is
issue #147.

## Current status

Engineering for the committed roadmap through Realtime is complete in the
stacked pull-request chain. Each phase remains isolated and reviewable so public
contracts, security-sensitive defaults, and release artifacts can be approved in
dependency order.

The stack must not be merged out of order or published by bypassing protected
OIDC release controls.

## 0.8.0 — platform foundation

Repository implementation and the release candidate are complete.

- [x] #132 — make optional integrations truly optional.
- [ ] #133 — verify the PyPI Trusted Publisher and protected GitHub `pypi`
      environment. The OIDC-only repository workflow is prepared; owner-side
      configuration remains.
- [x] #134 — replace duplicated feature documentation with a capability matrix.
- [x] #135 — execute supported examples and package smoke tests from the wheel.
- [x] #136 — define confidential vulnerability reporting and security policy.
- [ ] #137 — approve, merge, publish, and verify 0.8.0.

The 0.8.0 release candidate includes:

- the completed Codex plugin protocol, lifecycle, discovery, metadata,
  inspection, CLI, compatibility policy, and migration guidance;
- minimal `core`, `extract`, `ui`, and `all` installation profiles;
- OIDC-only PyPI publication with build-only rehearsal, SBOM, and attestations;
- canonical capabilities, installation, security, and release documentation;
- Python 3.10–3.13, minimum/latest SDK, clean-install, and installed-wheel gates.

Publication remains fail-closed until #133 records owner verification and #137
receives explicit human release approval.

## 0.8.x — shared runtime contracts

Engineering is complete in stacked PRs #154 and #156.

- [x] #138 — add a typed operation context and observability hooks.
- [x] #139 — define Responses continuation and Agents session semantics.

The implementation reuses official SDK tracing and state mechanisms rather than
building a parallel platform. The public contracts remain stacked behind the
0.8.0 release boundary.

## 0.9.0 — retrieval consolidation

Engineering is complete in stacked PRs #158, #160, and #162.

- [x] #140 — define one public retrieval API and migration map.
- [x] #141 — implement typed file and vector-store lifecycle helpers.
- [x] #142 — add File Search configuration and normalized results.

The resulting surface separates resource lifecycle, direct vector-store search,
hosted File Search configuration, and message composition. Existing imports
remain available, raw SDK resources are preserved, and remote cleanup is always
explicit.

## 0.10.0 — MCP

Engineering is complete in stacked PRs #163 and #164.

- [x] #143 — add typed hosted and Streamable HTTP MCP integration.
- [x] #144 — add filtering, fail-closed approvals, caching, bounded safe retries,
      and failure isolation.

MCP import, construction, connection, discovery, approval, execution, retry, and
cleanup remain explicit. Unknown or mutating tools are not silently approved or
retried. Final public transport and security policy still require human review.

## 0.11.0 — Realtime

Engineering is complete in stacked PRs #165 and #170.

- [x] #145 — add typed server-side session configuration and lifecycle helpers.
- [x] #146 — add normalized events, explicit tool execution, interruption,
      cancellation, and deterministic testing support.

The scope excludes browser UI, audio-device management, encoding, playback,
protocol reimplementation, hidden reconnect, and automatic tool execution. Raw
official runner, session, event, tool-call, result, and exception access remains
available. Final public and safety contracts still require human review.

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

Do not weaken a gate, merge the stack out of dependency order, or claim a
release before external owner controls and publication verification are
complete.
