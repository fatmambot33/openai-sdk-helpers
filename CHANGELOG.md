# Changelog

All notable changes to this project are documented here.

## Unreleased

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
