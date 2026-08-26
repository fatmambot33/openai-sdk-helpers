# Release notes

## 0.9.2 — 2026-08-26

Version 0.9.2 is a dependency-compatibility patch for installations that enable
LangExtract. It keeps pandas 3 out of the extraction dependency surface without
adding pandas to the core package or the Streamlit-only UI profile.

### Dependency compatibility

- `openai-sdk-helpers[extract]` now constrains pandas to `<3` alongside
  LangExtract, whose dependency metadata otherwise permits pandas 3.x.
- The same constraint is applied to the `dev` and `all` profiles because they
  include LangExtract.
- The core package remains pandas-free, and `openai-sdk-helpers[ui]` remains a
  Streamlit-only extra.

### Repository scope

- Current guidance and governance metadata are narrowed to thin OpenAI API/SDK
  helpers; abandoned general-purpose protocol, transport, discovery, trust, and
  policy surfaces are no longer treated as current product scope.

### Compatibility

No public Python symbols are removed. Python 3.10–3.13 and OpenAI Python
`>=2.45.0,<4.0.0` remain supported.

### Upgrade

```bash
pip install --upgrade openai-sdk-helpers==0.9.2
```

## 0.9.1 — 2026-08-25

Version 0.9.1 is a corrective patch for the 0.9 retrieval surface based on
delayed post-merge review of #162. It adds no new product scope and does not
change the 0.8.x release line.

### Retrieval correctness

- Direct vector-store search now sends the same normalized query values exposed
  through `RetrievalSearchPage.query`; blank sequence entries are removed before
  SDK execution.
- More than five normalized direct-search queries fail locally before either a
  synchronous or asynchronous SDK request.
- Strict result normalization now requires an actual filename rather than
  substituting the file identifier; lenient mode omits malformed results.
- Result attributes are validated against `Mapping[str, AttributeValue]` with
  string keys and scalar string, boolean, integer, or float values. Malformed
  attributes raise in strict mode and cause the result to be omitted in lenient
  mode.

### API and documentation compliance

- Package-internal retrieval imports now use the repository-required relative
  import style.
- Public search filters, adapters, and search mixins now include the required
  NumPy-style method and function documentation.
- Added deterministic sync and async regression tests for every behavioral
  delayed-review finding.

### Compatibility

No public symbols were removed, no dependencies were added, and legacy retrieval
imports are unchanged. Python 3.10–3.13 and OpenAI Python
`>=2.45.0,<4.0.0` remain supported.

### Upgrade

```bash
pip install --upgrade openai-sdk-helpers==0.9.1
```

## 0.9.0 — 2026-08-25

Version 0.9.0 consolidates retrieval around one typed, SDK-first public surface
for OpenAI Files, Vector Stores, direct vector-store search, and hosted File
Search. It builds on the 0.8.1 shared runtime contracts without hiding official
SDK clients, resources, identifiers, errors, filters, ranking controls, or raw
results.

### Retrieval lifecycle

- Added dependency-injected `OpenAIRetrievalClient` and
  `AsyncOpenAIRetrievalClient` lifecycle operations over caller-configured
  official OpenAI clients.
- Added explicit Files upload/delete and Vector Stores create, retrieve, list,
  update, and delete operations.
- Added existing-file attachment and upload-and-poll with explicit polling
  configuration and terminal-state validation.
- Treats only `completed` ingestion as successful; failed and cancelled terminal
  resources remain inspectable with their raw SDK state.
- Keeps detach distinct from deleting the underlying Files resource.
- Preserves caller ownership: injected clients and caller-owned file handles are
  never closed, and remote resources are never cleaned up implicitly.
- Batch uploads preserve input order and ordinary per-item exceptions while task
  cancellation and process interrupts propagate.

### Direct vector-store search

- Added matching synchronous and asynchronous `search()` methods.
- Supports one query or an ordered sequence of queries, common typed attribute
  filters, explicit official SDK filter mappings, result limits, ranking
  options, and query rewriting.
- Normalizes file identity, score, attributes, content fragments, pagination,
  and query values while retaining the raw SDK page and each raw result.
- Strict normalization raises indexed errors for malformed SDK-shaped items;
  lenient mode omits malformed items without reordering valid results.
- Empty result sets are valid and deterministic.

### Hosted File Search adapters

- Added `FileSearchConfig` adapters for Responses request mappings and the
  official Agents SDK `FileSearchTool`.
- Responses adaptation copies caller input rather than mutating it and adds
  `file_search_call.results` only when explicitly requested.
- Included File Search tool-call results can be normalized without issuing a
  second request.
- File citation annotations can be collected in model-output order while
  preserving the raw annotation object.
- Hosted-tool configuration remains separate from direct search execution and
  does not upload files, create vector stores, issue model requests, or assume
  cleanup ownership.

### Compatibility and migration

Existing `FilesAPIManager`, `VectorStorage`, response file/vector-store helpers,
and Agents file-message builders remain available. The 0.9 retrieval package is
the preferred surface for new code, but this release does not remove legacy
imports or silently migrate ownership semantics.

The package continues to support Python 3.10–3.13, OpenAI Python
`>=2.45.0,<4.0.0`, and the existing optional `core`, `extract`, `ui`, and `all`
installation profiles. The final retrieval heads passed the minimum/latest SDK
matrix, installed-wheel smoke tests, security checks, and deterministic
network-free tests.

Review [docs/retrieval.md](docs/retrieval.md),
[docs/retrieval-lifecycle.md](docs/retrieval-lifecycle.md), and
[docs/file-search.md](docs/file-search.md) for migration boundaries, ownership,
search behavior, filters, citations, and raw SDK escape hatches.

### Upgrade

```bash
pip install --upgrade openai-sdk-helpers==0.9.0
```

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

`openai-sdk-helpers` now supports OpenAI Python `>=2.45.0,<4.0.0`, including the
3.x line. The compatibility matrix, full Python 3.10–3.13 test suite, clean
installation profiles, and installed-wheel smoke tests pass with the latest 3.x
SDK. OpenAI Python 3.x changes its default HTTP client to HTTPX2, so applications
that construct custom HTTPX clients or transports should follow the official
OpenAI SDK migration guidance.

The package now explicitly declares `tqdm`, which is required by the public
`VectorStorage` batch progress surface and was previously missing from wheel
runtime metadata.

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
