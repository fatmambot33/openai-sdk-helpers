# Security policy

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability.

Use GitHub private vulnerability reporting for this repository:

1. Open the repository **Security** tab.
2. Select **Advisories**.
3. Select **Report a vulnerability**.

This creates a confidential security advisory visible only to the reporter and
repository security maintainers. Private vulnerability reporting is enabled for
this repository.

Do not include real API keys, access tokens, customer prompts, model responses,
uploaded files, personal data, or production logs. Revoke and rotate any secret
that may already have been exposed before submitting the report.

A useful report includes:

- the affected package version and commit, when known;
- Python version and operating system;
- the smallest safe reproducer using placeholder credentials and synthetic data;
- expected and observed behavior;
- security impact and the conditions required to trigger it;
- whether credentials, files, plugin commands, tools, or external services are involved;
- suggested remediation, when available.

The maintainers will acknowledge the report, investigate its scope, coordinate a
fix and disclosure plan when appropriate, and publish an advisory after affected
users have a reasonable upgrade path. Response and remediation timing depends on
severity, reproducibility, maintainer availability, and upstream SDK behavior;
this project does not promise fixed response-time targets.

## Supported versions

Until `0.8.0` is published, the latest `0.7.x` release is the supported release
line. After `0.8.0` is published, support moves to the latest `0.8.x` release and
older `0.7.x` releases no longer receive routine security fixes.

| Release line | Security support |
| --- | --- |
| Latest published minor line | Supported |
| Older minor lines | Upgrade required |
| Unreleased development branches | Best effort; not a supported release |

Security fixes are normally released in the newest compatible patch or minor
release. A severe vulnerability may require a breaking change when a safe
backward-compatible fix is not possible; such a decision requires explicit
human review and migration guidance.

## Security boundaries

### Credentials and configuration

- API keys and tokens must come from caller-owned environment or secret stores.
- The package must not log, serialize, commit, or persist credentials by default.
- Examples and tests must use placeholders only.
- Diagnostic output must redact secrets and avoid printing complete
  authorization headers.
- A credential exposed in an issue, log, commit, artifact, or screenshot must be
  revoked rather than merely deleted from the visible surface.

### Prompts, responses, and tracing

Prompts, tool arguments, model responses, uploaded content, and trace data may be
sensitive. New diagnostics or observability features must exclude content by
default or require explicit caller opt-in, document retention behavior, and
preserve access to underlying official SDK controls.

### Files and vector stores

File upload, download, search, deletion, and cleanup operations must make network
calls and destructive actions explicit. Helpers must not automatically delete
caller-owned resources or expose file contents in logs. Tests use synthetic data
and mocked network boundaries.

### Plugins and tools

Codex plugins and tool handlers execute caller-provided code. Discovery must not
silently execute commands. Plugin startup, tool invocation, approvals, failure
isolation, and shutdown must remain explicit. Unknown or mutating tools must not
be granted broader trust merely because they were discovered successfully.

### Optional and external integrations

Optional integrations remain outside the base installation. Realtime, UI,
extraction, and future external-service adapters must define trust boundaries,
resource ownership, cancellation, failure behavior, and credential handling
before becoming public capabilities.

### Dependency and supply-chain changes

Dependency-bound changes, release workflows, artifact attestations, publishing
identity, and credential migration require review against
[docs/release-checklist.md](docs/release-checklist.md). Package publication uses
PyPI Trusted Publishing and must not fall back to a long-lived PyPI API token.

## Public discussion

After a fix or advisory is public, normal discussion may continue in issues or
pull requests. Before disclosure, keep exploit details, affected deployments,
private patches, and reproducer material inside the private security advisory.
