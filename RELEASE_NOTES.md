# Release notes

## 0.8.0 — 2026-08-06

Version 0.8.0 completes the Codex plugin roadmap and establishes the package's
production foundation: optional integration profiles, secure publishing,
canonical capability documentation, executable installed-wheel examples, and a
confidential security policy.

### Codex plugin foundation and hardening

- Added the typed `CodexPlugin`, `CodexCommand`, `CodexPluginContext`, and
  `CodexPluginRegistry` surface.
- Added deterministic plugin and command registration.
- Added installed-package discovery through the `openai_sdk_helpers.codex`
  entry-point group.
- Added startup and shutdown lifecycle hooks, asynchronous commands, and atomic
  rollback when setup fails.
- Added optional `CodexPluginMetadata` with contract version, implementation
  version, summary, capabilities, and deprecation state.
- Added deterministic capability inspection and isolated discovery reports that
  preserve healthy plugins when another entry point fails.
- Added `openai-helpers codex plugins` and `openai-helpers codex commands`.
- Added installed entry-point end-to-end tests, compatibility policy, packaging
  guidance, and migration documentation.

Existing plugins implementing only `name` and `setup(context)` remain compatible.
Fail-fast `discover()` behavior is unchanged; isolated discovery is an explicit
opt-in for hosts loading optional or third-party packages.

### Minimal core and optional integrations

LangExtract and Streamlit are no longer mandatory dependencies.

- `openai-sdk-helpers` installs the core SDK-first toolkit.
- `openai-sdk-helpers[extract]` enables LangExtract-backed structures, agents,
  and extraction helpers.
- `openai-sdk-helpers[ui]` enables Streamlit application helpers.
- `openai-sdk-helpers[all]` enables both optional capabilities.

Extraction imports remain backward-compatible when the `extract` profile is
installed. Missing optional dependencies now raise errors containing the exact
installation command. CI validates clean `core`, `extract`, `ui`, and `all`
installations independently.

### Release and supply-chain controls

- Removed the legacy `PYPI_API_TOKEN` fallback.
- Made PyPI Trusted Publishing through GitHub OIDC the only publication path.
- Added a build-only manual rehearsal that validates distributions without
  publishing or creating a release.
- Build artifacts are created once, metadata-checked, installed in a clean
  environment, given a reproducible CycloneDX SBOM, and attested before the
  minimal publish job receives them.
- Added documented publisher identity, protected-environment requirements,
  verification, and recovery procedures.

### Executable package contract

- Added a package-smoke workflow that installs the exact built wheel.
- Verifies `py.typed`, bundled Jinja templates, and both CLI entry points.
- Executes supported Responses, Agents, and Codex examples without credentials,
  `PYTHONPATH`, paid API calls, or external network access.
- Classified examples as supported, illustrative, or deprecated.

### Documentation and security

- Added a canonical capability matrix covering maturity, installation,
  execution style, official SDK relationship, and escape hatches.
- Replaced duplicated README inventories with concise navigation.
- Added deterministic internal Markdown path and anchor validation.
- Added confidential GitHub private vulnerability reporting guidance,
  supported-version policy, and explicit security boundaries.
- Added a release checklist covering compatibility, credentials, publishing,
  files, plugins, tools, transports, diagnostics, artifacts, and post-release
  verification.

### Compatibility note

`ResponseConfiguration` now validates `tools` at initialization. The value must
be a non-string sequence of mapping objects. String-like containers and
non-mapping entries fail immediately with `TypeError`. Valid official SDK-shaped
tool dictionaries require no change.

### Upgrade

```bash
pip install --upgrade openai-sdk-helpers==0.8.0
```

Review [docs/migration-0.8.md](docs/migration-0.8.md) before upgrading extraction
or Streamlit installations. Plugin maintainers should also review
[docs/codex-plugin-migration-0.8.md](docs/codex-plugin-migration-0.8.md).
