# Changelog

All notable changes to this project are documented here.

## Unreleased

## 0.8.1 - 2026-08-25

- Added the optional typed `OperationContext` lifecycle shared by Responses, Agents, and Codex commands, with vendor-neutral observers, usage capture, explicit retry metadata, and safe diagnostics that redact sensitive content by default.
- Added explicit conversation-state ownership through `AgentRunState` and `ResponseContinuation`, rejecting ambiguous combinations before SDK execution while preserving the existing `session=` shorthand.
- Added caller-owned `LocalMessageStore` persistence with explicit save, resume, clear, close-without-save, and delete semantics.
- Preserved original SDK results, exceptions, identifiers, tracing, and session objects without adding a telemetry backend or hidden state selection.

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
- Classified examples as supported, illustrative, or deprecated with an executable compatibility policy.
- Added a canonical capability matrix covering maturity, installation, execution, and SDK escape hatches.
- Replaced the duplicated README feature inventory with concise canonical documentation navigation.
- Added deterministic internal Markdown file and anchor validation to CI.
- Removed the legacy PyPI API-token publishing fallback in favor of OIDC-only Trusted Publishing.
- Added a non-publishing release rehearsal mode and documented publisher setup, verification, and recovery.
- Made LangExtract and Streamlit optional through the `extract`, `ui`, and `all` installation profiles.
- Added lazy extraction exports with actionable missing-extra errors and clean-install CI coverage.
- Completed Codex plugin production hardening.
- Added optional structured plugin metadata and capability inspection.
- Added isolated entry-point discovery reports without changing fail-fast discovery.
- Added `openai-helpers codex plugins` and `openai-helpers codex commands`.
- Added installed entry-point discovery coverage, compatibility policy, and a 0.8 migration guide.
- Standardized repository documentation.
