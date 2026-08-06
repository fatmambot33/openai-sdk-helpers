# Changelog

All notable changes to this project are documented here.

## Unreleased

- Added explicit `AgentRunState`, `ResponseContinuation`, and `LocalMessageStore` contracts for Agents sessions, server-managed continuation, and package-local message persistence.
- Added early validation for incompatible state mechanisms while preserving existing stateless calls, raw SDK identifiers, underlying session objects, and the legacy `session=` shorthand.
- Added an optional typed `OperationContext` with sync and async lifecycle hooks across Responses runners, Agents runners, and Codex command execution.
- Added vendor-neutral usage capture and safely redacted diagnostics while preserving original SDK results and exceptions.

## 0.8.0 - 2026-08-06

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
- Added installed entry-point discovery coverage, compatibility policy, and 0.8 migration guides.
- Standardized repository documentation.
