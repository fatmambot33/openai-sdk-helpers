# Release Notes

## Unreleased

### Breaking Changes (beta)

- `ResponseConfiguration` now performs strict `tools` validation at initialization time.
  - `tools` must be a non-string sequence of mapping objects.
  - String-like containers (`str`, `bytes`, `bytearray`) are rejected.
  - Non-mapping tool items raise `TypeError` immediately.
- `add_web_search_tool=True` now reliably appends a raw `{"type": "web_search"}` tool definition that is compatible with `ResponseBase` request construction.
