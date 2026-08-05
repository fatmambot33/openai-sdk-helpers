# Release Notes

## Unreleased

### Codex plugin hardening

- Added optional `CodexPluginMetadata` with contract version, implementation version, summary, capabilities, and deprecation state.
- Added deterministic plugin and command inspection through `inspect_plugins()`.
- Added isolated installed-plugin discovery through `discover_isolated()` with structured failure reports.
- Added `openai-helpers codex plugins` and `openai-helpers codex commands`.
- Added compatibility policy and migration guidance for the next minor release.

These additions are backward compatible. Existing plugins that only implement `name` and `setup(context)` continue to work unchanged, and `discover()` retains fail-fast behavior.

### Breaking Changes (beta)

- `ResponseConfiguration` now performs strict `tools` validation at initialization time.
  - `tools` must be a non-string sequence of mapping objects.
  - String-like containers (`str`, `bytes`, `bytearray`) are rejected.
  - Non-mapping tool items raise `TypeError` immediately.
- `add_web_search_tool=True` now reliably appends a raw `{"type": "web_search"}` tool definition that is compatible with `ResponseBase` request construction.
