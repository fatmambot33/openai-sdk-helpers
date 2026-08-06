# Publishing

`openai-sdk-helpers` publishes through PyPI Trusted Publishing. The release
workflow never accepts a PyPI username, password, or long-lived API token.

## Trust configuration

Configure the existing PyPI project with this GitHub Actions publisher:

| Field | Value |
| --- | --- |
| PyPI project | `openai-sdk-helpers` |
| GitHub owner | `fatmambot33` |
| GitHub repository | `openai-sdk-helpers` |
| Workflow filename | `python-publish.yml` |
| GitHub environment | `pypi` |

On PyPI, open the project management page, select **Publishing**, add a GitHub
Actions Trusted Publisher, and enter the values above. The filename is the
basename under `.github/workflows`, not the complete path.

On GitHub, create the `pypi` environment and protect it with required reviewers.
Restrict deployment branches to `main`. The environment does not need a PyPI
secret: the publish job receives only `id-token: write`, exchanges its GitHub
OIDC identity for a short-lived PyPI credential, and uploads the validated
artifacts.

After one successful Trusted Publishing release, remove any legacy
`PYPI_API_TOKEN` repository or environment secret. The workflow does not read
that secret and intentionally has no token fallback.

## Release flow

A production publication is built once and reused throughout the workflow:

1. Resolve the package version and check whether it already exists on PyPI.
2. Build the wheel and source distribution in an isolated job.
3. Validate metadata with Twine.
4. Install and import the exact wheel in a clean virtual environment.
5. Generate a reproducible CycloneDX SBOM.
6. Create GitHub artifact attestations for the distributions.
7. Pass the artifacts to the minimal `pypi` environment job.
8. Publish through `pypa/gh-action-pypi-publish` using OIDC.
9. Create or reconcile the matching `v<version>` GitHub release from the same
   commit and artifacts.

The PyPA publish action also uploads PyPI publish attestations by default when
Trusted Publishing is used.

## Rehearsal without publishing

Run **Release Python package** manually and leave **Publish to PyPI and reconcile
the GitHub release** disabled. The workflow performs version resolution, package
build, metadata validation, clean-wheel installation, SBOM generation, artifact
attestation, and artifact upload. It skips both PyPI publication and GitHub
release creation.

Download the `python-package-distributions` and `release-evidence` workflow
artifacts to inspect the exact release candidates. This is the supported dry-run
path and does not require a TestPyPI project or credential.

For a local equivalent:

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m venv .venv-release-check
.venv-release-check/bin/python -m pip install dist/*.whl
.venv-release-check/bin/openai-helpers --help
```

## Production release

1. Merge a version and changelog update into `main` after all repository checks
   pass.
2. Review the triggered release workflow.
3. Approve the `pypi` environment deployment.
4. Confirm the publish job succeeds.
5. Verify the PyPI project shows the expected version and attestations.
6. Verify the matching GitHub release contains the same distributions and SBOM.
7. Install the released version from PyPI in a clean environment and run the CLI
   smoke checks.

A manual workflow run can also publish when the `publish` input is explicitly
enabled and the `pypi` environment deployment is approved.

## Recovery

Trusted Publishing failures should not be bypassed with a token.

- For `invalid-publisher`, compare the PyPI publisher owner, repository,
  workflow filename, and environment with the table above.
- If the repository, workflow, or environment is renamed, update the PyPI
  publisher before retrying.
- If a release version already exists on PyPI, do not overwrite it. Correct the
  source, increment the version, and publish a new immutable release.
- If PyPI publication succeeds but GitHub release reconciliation fails, rerun
  the workflow. The release job detects the existing immutable PyPI version and
  reconciles the matching tag and release without uploading the package again.
- If the Trusted Publisher must be replaced, register the new publisher first,
  validate it with a new release, and then remove the obsolete publisher.

## Human-owned controls

The following actions require a repository or PyPI owner:

- registering or changing the PyPI Trusted Publisher
- creating and protecting the GitHub `pypi` environment
- approving a production deployment
- removing the legacy PyPI API token
- changing publisher identity fields

Repository automation may prepare and validate release artifacts, but it must
not weaken these controls or publish without explicit environment approval.
