# Release checklist

Use this checklist for every public package release. Release publication and
security-sensitive decisions require explicit human approval even when agents
prepare the code, artifacts, and release notes.

## Scope and compatibility

- [ ] The release contains only approved issue scope.
- [ ] Public additions satisfy the feature acceptance test in `PRODUCT.md`.
- [ ] Breaking changes and deprecations have explicit migration guidance.
- [ ] `docs/capabilities.md`, `docs/public-api.md`, focused guides, README, and
      changelog agree with the shipped surface.
- [ ] Optional integrations remain outside the base installation.
- [ ] Underlying OpenAI SDK clients, resources, results, events, and exceptions
      remain available where promised.

## Security review

Require named human review when a release changes any of the following:

- credentials, environment loading, authentication, or secret storage;
- package publishing, GitHub Actions permissions, OIDC identity, attestations,
  artifacts, or dependency bounds;
- file upload, download, vector-store lifecycle, deletion, or cleanup;
- plugin discovery, startup, shutdown, command execution, or entry points;
- tool execution, approvals, MCP servers, external transports, or network trust;
- prompt, response, trace, diagnostic, log, or telemetry content;
- Realtime sessions, cancellation, reconnection, audio buffers, or event handling;
- sandboxing, subprocesses, filesystem access, or serialization of untrusted data.

For an applicable change, confirm:

- [ ] credentials and authorization headers are never logged or serialized;
- [ ] examples, tests, fixtures, screenshots, and documentation use placeholders;
- [ ] sensitive prompts, responses, files, and tool arguments are excluded from
      diagnostics by default;
- [ ] destructive actions require explicit caller intent;
- [ ] resource ownership, cleanup, cancellation, and failure isolation are documented;
- [ ] unknown plugins or tools are not silently trusted or executed;
- [ ] dependency and workflow permissions are minimal and justified;
- [ ] vulnerability reporting instructions remain accurate and confidential.

## Automated validation

- [ ] `pydocstyle src` passes.
- [ ] `black --check --diff .` passes.
- [ ] `pyright src` passes.
- [ ] tests pass with the configured coverage threshold.
- [ ] Python 3.10–3.13 pass.
- [ ] minimum and latest supported OpenAI SDK dependency sets pass.
- [ ] internal Markdown links and anchors pass.
- [ ] `core`, `extract`, `ui`, and `all` clean-install profiles pass.
- [ ] wheel and source distributions build and pass metadata validation.
- [ ] the exact wheel installs in a clean environment.
- [ ] both CLI entry points and supported examples run without credentials.
- [ ] required runtime package data and `py.typed` are present.

## Release artifacts

- [ ] The canonical version is updated consistently.
- [ ] The changelog has a dated release section.
- [ ] Release notes explain user-visible changes, compatibility, and migration.
- [ ] The wheel and source distribution are built once and reused.
- [ ] The validated artifacts have a CycloneDX SBOM and GitHub attestations.
- [ ] PyPI Trusted Publisher identity matches the documented owner, repository,
      workflow, and protected environment.
- [ ] No `PYPI_API_TOKEN` fallback is enabled.
- [ ] The GitHub release and PyPI publication use the same commit and artifacts.

## Publication and verification

- [ ] A human approves the protected `pypi` environment deployment.
- [ ] PyPI shows the intended immutable version and attestations.
- [ ] The matching `v<version>` GitHub release contains the distributions and SBOM.
- [ ] A clean environment installs the version from PyPI.
- [ ] package import, `openai-helpers --help`, and
      `openai-helpers-credentials --help` succeed after publication.
- [ ] The release issue records validation evidence and any approved exception.

Do not publish when a required item is unknown. Record the exact blocker instead
of weakening a check, bypassing OIDC, inventing an approval, or claiming a gate
is complete.
