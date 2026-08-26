# Security enforcement evidence

Security scanning for the default branch is enforced by the repository ruleset rather than by a duplicate repository workflow.

The active `Main protect` ruleset (repository ruleset ID `20521246`) requires GitHub CodeQL code scanning on the default branch with security alerts at `high_or_higher` and analysis alerts at `errors`.

The repository intentionally does not duplicate that enforcement in `.github/workflows/security.yml`. This keeps GitHub Actions lightweight while preserving the AI-native platform requirement that security scanning is mandatory and independently enforced.

See `SECURITY.md` for vulnerability reporting, credential boundaries, and supply-chain policy.
