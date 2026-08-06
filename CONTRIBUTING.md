# Contributing

Security vulnerabilities must not be reported in public issues. Follow the
confidential process in [SECURITY.md](SECURITY.md).

## Workflow

1. Open or select one focused issue.
2. Branch from the current `main` branch.
3. Keep the implementation small enough for one coherent pull request.
4. Add or update deterministic tests.
5. Add type hints and NumPy-style docstrings.
6. Update the applicable canonical documentation.
7. Run all repository checks.
8. Open a pull request with scope, compatibility impact, validation, and issue links.

Before merging, confirm the change supports [PRODUCT.md](PRODUCT.md), preserves a
usable core installation, and does not hide OpenAI API calls or resource
ownership behind surprising defaults.

## Required checks

```bash
pydocstyle src
black --check --diff .
pyright src
pytest -q --cov=src --cov-report=term-missing --cov-fail-under=70
python scripts/check_markdown_links.py
```

CI additionally validates supported Python versions, minimum/latest compatible
OpenAI SDK dependencies, built distributions, installed package profiles, and
repository governance.

## Documentation ownership

A public change must update every applicable canonical document:

- [docs/capabilities.md](docs/capabilities.md) for capability, maturity,
  installation, execution, or escape-hatch changes;
- [docs/public-api.md](docs/public-api.md) for intentional import changes;
- [docs/installation.md](docs/installation.md) for dependency profiles;
- the focused behavior guide for semantics and examples;
- [SECURITY.md](SECURITY.md) for trust-boundary or supported-version changes;
- [docs/release-checklist.md](docs/release-checklist.md) for release-gate changes;
- [CHANGELOG.md](CHANGELOG.md) for user-visible changes;
- [README.md](README.md) only when top-level navigation or the product summary changes.

Do not copy the capability matrix into the README or repeat long guides across
multiple files. Link to the canonical source instead. Internal Markdown links
must pass the repository link checker.

## Public API changes

Public additions require:

- an issue that explains the concrete user workflow;
- a typed, documented contract;
- sync/async semantics where applicable;
- explicit ownership and cleanup behavior;
- access to underlying official SDK objects;
- compatibility and migration notes;
- deterministic success and failure tests.

Breaking changes, deprecations, security-sensitive defaults, releases, and new
public contracts require human review even when implementation is automated.

## Security-sensitive changes

Use the security section of [docs/release-checklist.md](docs/release-checklist.md)
when changing credentials, publishing, files, plugins, tools, external
transports, diagnostics, tracing, subprocesses, or destructive resource
operations. Pull requests must explain trust boundaries, redaction, ownership,
cleanup, cancellation, and failure isolation where applicable.

Examples, tests, fixtures, screenshots, and documentation must contain
placeholder credentials and synthetic data only. Do not copy production logs,
prompts, responses, files, or tool arguments into a pull request.

## Pull-request completion

A pull request is ready to merge only when:

- all required checks pass on the final head commit;
- no unresolved review thread remains;
- documentation and changelog entries match the implementation;
- optional integrations remain outside the base installation;
- no credential, paid API call, or external network dependency is required by
  pull-request tests;
- applicable security review items are completed by a human reviewer;
- the linked issue's acceptance criteria are genuinely satisfied.
