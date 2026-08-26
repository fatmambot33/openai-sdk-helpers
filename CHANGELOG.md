# Changelog

All notable changes to this project are documented here.

## Unreleased

- Narrowed the product scope to thin OpenAI API/SDK helpers and removed abandoned general-purpose protocol, transport, discovery, trust, and policy surfaces from current repository guidance and governance metadata.

## 0.9.1 - 2026-08-25

- Harden direct vector-store search normalization by forwarding only normalized queries, rejecting more than five direct-search queries before the SDK call, requiring real filenames in strict mode, and validating result attribute keys and scalar values.
- Complete NumPy-style documentation for the public File Search adapters, filters, and search mixins, and keep package-internal runtime imports relative.

## 0.9.0 - 2026-08-25

- Added dependency-injected synchronous and asynchronous retrieval clients for explicit Files and Vector Stores lifecycle operations.
- Added ordered partial-failure batch uploads, SDK-backed polling configuration, attachment status, explicit detach-versus-delete behavior, and optional operation observability.
- Added direct synchronous and asynchronous vector-store search with explicit filters, ranking options, query rewriting, pagination metadata, strict or lenient normalization, and raw SDK escape hatches.
- Added hosted File Search adapters for Responses and the Agents SDK, including included-result normalization and ordered file-citation extraction without hidden requests.
- Added typed common comparison and compound attribute filters while retaining explicit official SDK mapping escape hatches.
- Added a migration map separating resource lifecycle, search configuration, and message composition while preserving legacy retrieval imports for compatibility.

## 0.8.1 - 2026-08-25

- Added the optional typed `OperationContext` lifecycle shared by Responses, Agents, and Codex commands, with vendor-neutral observers, usage capture, explicit retry metadata, and safe diagnostics that redact sensitive content by default.
- Added explicit conversation-state ownership through `AgentRunState` and `ResponseContinuation`, rejecting ambiguous combinations before SDK execution while preserving the existing `session=` shorthand.
- Added caller-owned `LocalMessageStore` persistence with explicit save, resume, clear, close-without-save, and delete semantics.
- Preserved original SDK results, exceptions, identifiers, tracing, and session objects without adding a telemetry backend or hidden state mechanism.

## 0.8.0 - 2026-08-06

- Established the 0.8 compatibility line with consolidated migration guidance for optional extraction and UI profiles, Codex plugins, tool validation, and OIDC publishing.
- Expanded OpenAI Python compatibility to `>=2.45.0,<4.0.0`, including validated 3.x support, and declared the `tqdm` runtime dependency required by the public vector-storage surface.
- Aligned the Python distribution and bundled Codex plugin manifest at version 0.8.0.
- Carried forward the production foundation shipped on the 0.7.5 branch without duplicating its detailed release notes.

## 0.7.5 - 2026-08-06

- Added a security policy using confidential GitHub private vulnerability reporting and an explicit supported-version transition.
- Added a security-aware release checklist covering credentials, publishing, files, plugins, tools, transports, diagnostics, and artifact verification.
- Replaced a secret-shaped example value with an explicit non-secret placeholder.
- Added built-wheel smoke tests for both CLI entry points, runtime package data, and credential-free Responses, Agents, and Codex examples.