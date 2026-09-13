# AI Native Platform — Inheritance Implementation Plan

## Objective

Turn **AI Native Platform** from a collection of repository conventions into an **inheritable platform contract**.

Derived repositories should inherit common AI-native behavior instead of copying framework files and instructions.

The initial inheritance kernel is:

- `welcome`
- `troubleshooting`
- `update`
- `doctor`

The target relationship is:

```text
AI Native Platform
        │
        ├── Core contract
        ├── Shared capabilities
        ├── Inherited skills
        │   ├── welcome
        │   ├── troubleshooting
        │   └── update
        └── doctor
                │
                ▼
             Profile
                │
                ▼
        Derived repository
                │
                └── Local extensions
```

The fundamental rule is:

> Derived repositories extend AI Native Platform. They do not copy it.

---

## 0. Preconditions

Do not start implementation until the repository tree is clean.

Run:

```bash
git status
git fetch --all --prune
git pull --ff-only
```

Required state:

- [ ] Working tree is clean.
- [ ] Current branch is up to date.
- [ ] CI is green.
- [ ] No unresolved PR is modifying the same platform architecture.
- [ ] No release work is pending that should land first.
- [ ] Existing Codex review comments affecting this work are resolved.

Create a dedicated branch:

```bash
git switch -c feat/ai-native-inheritance
```

### Codex review policy

Codex review is asynchronous and quota-limited.

For this implementation:

- Run deterministic tests and static checks continuously.
- Do not request Codex review after every incremental change.
- Batch related fixes.
- Request Codex review only at meaningful merge-ready checkpoints.
- Do not repeatedly request review of substantially identical HEADs.
- Resolve useful review findings before requesting another review.
- Confirm CI is green on the current HEAD before using Codex as a merge gate.

---

## 1. Architectural principles

### 1.1 Inheritance over duplication

Shared behavior belongs in the platform.

Derived repositories contain only:

- project identity;
- repository-specific context;
- project-specific validation;
- project-specific workflows;
- local skill extensions;
- explicit overrides.

Do not copy shared platform skills into every repository.

### 1.2 Composition over replacement

Resolution order:

```text
Platform defaults
        ↓
Profile
        ↓
Repository configuration
        ↓
Repository extensions
        ↓
Explicit override
```

A repository should normally extend inherited behavior.

Example:

```text
Base troubleshooting
        +
Python-library troubleshooting
        +
PermutiveAPI troubleshooting
```

Full replacement must be explicit and should produce a `doctor` warning.

### 1.3 Skills orchestrate capabilities

Do not implement large monolithic prompt files.

Reusable capabilities should perform repository operations. Skills should orchestrate those capabilities.

```text
welcome
 ├── inspect_repository
 ├── inspect_ai_native_state
 └── inspect_validation

troubleshooting
 ├── inspect_repository
 ├── inspect_ci
 ├── inspect_dependencies
 └── run_validation

update
 ├── inspect_ai_native_state
 ├── report_platform_drift
 ├── apply_migration
 └── run_validation

doctor
 ├── inspect_ai_native_state
 ├── inspect_repository
 ├── inspect_ci
 └── run_validation
```

### 1.4 Updates must preserve local work

The update mechanism must understand ownership.

Artifacts need one of the following ownership modes:

```text
inherited
managed
merged
local
```

An update must never blindly regenerate repository-owned files.

### 1.5 Deterministic validation first

AI review is not a substitute for deterministic validation.

Every change must first pass:

- tests;
- type checking;
- lint/style checks;
- schema validation;
- inheritance tests;
- migration tests;
- doctor checks.

---

## 2. Define the inheritance contract

Introduce four architectural concepts.

### Platform

Owns:

- skill semantics;
- capability contracts;
- inheritance behavior;
- validation rules;
- versioning;
- migrations;
- platform defaults.

### Profile

Specializes the platform for a family of repositories.

Initial profile:

```text
python-library
```

Possible future profiles:

```text
python-application
typescript-library
web-application
service
android-application
data-project
```

Do not create profiles until real repositories require them.

### Derived repository

Contains:

- project metadata;
- project context;
- repository-specific validation;
- local extensions;
- explicit overrides.

### Local extension

Augments inherited behavior without copying the base implementation.

---

## 3. Introduce the AI Native manifest

Create:

```text
.ai-native/manifest.yaml
```

Initial schema:

```yaml
schema_version: 1

platform:
  name: ai-native-platform
  compatible: "^1.0"

profile:
  name: python-library
  compatible: "^1.0"

skills:
  inherit:
    - welcome
    - troubleshooting
    - update

capabilities:
  validation: true
  repository_inspection: true
  dependency_inspection: true
  release_inspection: true

validation:
  commands:
    - pytest
    - pyright
    - pydocstyle

update:
  strategy: managed

overrides: {}
```

The v1 manifest must answer:

1. What platform does this repository derive from?
2. Which platform versions are compatible?
3. Which profile applies?
4. Which skills are inherited?
5. What does the repository extend?
6. How is the repository validated?
7. How should updates be handled?

Avoid speculative fields.

---

## 4. Implement manifest validation

Add a formal schema and parser.

Required validation:

- [ ] Supported `schema_version`.
- [ ] Valid platform declaration.
- [ ] Valid compatibility constraint.
- [ ] Valid profile declaration.
- [ ] Known inherited skills.
- [ ] Known extension modes.
- [ ] Valid validation commands.
- [ ] Valid update strategy.
- [ ] Unknown fields handled intentionally.
- [ ] Clear errors for malformed configuration.

Tests must cover valid and invalid manifests.

---

## 5. Implement inheritance resolution

Create one deterministic resolver.

Resolution order:

```text
platform
→ profile
→ repository
→ repository extensions
→ explicit overrides
```

The resolver should preserve provenance.

For every effective value, it should eventually be possible to determine:

```text
value
source
ownership
override status
```

Example:

```text
troubleshooting
├── platform: generic failure diagnosis
├── profile: Python/package diagnostics
└── repository: Permutive-specific diagnostics
```

Required tests:

- [ ] Platform-only resolution.
- [ ] Platform + profile.
- [ ] Platform + profile + repository.
- [ ] Append semantics.
- [ ] Override semantics.
- [ ] Disable semantics.
- [ ] Conflict detection.
- [ ] Stable resolution ordering.

---

## 6. Define extension semantics

Support only a minimal v1 vocabulary:

```text
inherit
append
override
disable
```

Example extension:

```yaml
skills:
  troubleshooting:
    inherit: true
    append:
      - .ai-native/local/troubleshooting.md
```

Explicit override:

```yaml
skills:
  troubleshooting:
    override: .ai-native/local/custom-troubleshooting.md
```

Rules:

- `inherit` is the default.
- `append` is preferred for specialization.
- `override` is exceptional.
- `disable` must be explicit.
- `override` and `disable` of core inherited behavior should generate warnings in `doctor`.

---

## 7. Introduce reusable capabilities

Implement platform-level capabilities before building complex skills.

Initial capability set:

```text
inspect_repository
inspect_ai_native_state
inspect_validation
inspect_ci
inspect_dependencies
inspect_release_state
read_known_issues
run_validation
report_platform_drift
apply_migration
```

Capabilities must:

- have a clear contract;
- be independently testable;
- return structured results where practical;
- avoid embedding repository-specific assumptions.

---

## 8. Implement `welcome`

`welcome` is the standard entry point for an agent entering a repository.

It should answer:

- What is this repository?
- What does it do?
- What platform version applies?
- Which profile applies?
- What architecture matters?
- Which commands validate changes?
- How are releases handled?
- What should the agent avoid?
- Which inherited capabilities are available?
- Which repository-specific extensions exist?
- What should be read next?

Inputs should come from repository state:

```text
manifest
AGENTS.md
README
package metadata
repository structure
CI configuration
local extensions
```

The platform skill must not contain repository-specific facts.

Definition of done:

- [ ] Works on platform repository.
- [ ] Works on a minimal derived fixture.
- [ ] Correctly includes profile information.
- [ ] Correctly includes repository extensions.
- [ ] Does not require copied repository-specific skill content.

---

## 9. Implement `troubleshooting`

Use a fixed troubleshooting lifecycle:

```text
1. Classify failure
2. Collect evidence
3. Reproduce
4. Inspect known failure modes
5. Isolate root cause
6. Apply smallest safe fix
7. Run deterministic validation
8. Check regressions
9. Record reusable learning
```

Base failure categories:

```text
environment
dependency
configuration
test
typing
lint
CI
release
integration
platform inheritance
unknown
```

The base skill must remain technology-neutral.

Profiles may add specialized categories.

---

## 10. Implement `doctor`

`doctor` determines whether a repository correctly derives from AI Native Platform.

Initial checks:

- [ ] Manifest exists.
- [ ] Manifest schema is valid.
- [ ] Platform version resolves.
- [ ] Profile resolves.
- [ ] Required inherited skills resolve.
- [ ] Validation contract exists.
- [ ] Agent instructions are available.
- [ ] CI configuration exists where required.
- [ ] Repository extensions resolve.
- [ ] No inheritance conflicts exist.
- [ ] No unknown overrides exist.
- [ ] No deprecated platform fields are used.
- [ ] Platform drift can be determined.

Example output:

```text
AI Native Doctor

Platform             1.0
Profile              python-library 1.0

Manifest             PASS
Welcome              PASS
Troubleshooting      PASS
Update               PASS
Validation contract  PASS
CI                    PASS
Extensions            PASS
Platform drift        NONE

Status: compliant
```

Suggested exit codes:

```text
0 = compliant
1 = compliant with warnings
2 = non-compliant
3 = invalid configuration
```

CI should eventually be able to run:

```bash
ai-native doctor --ci
```

---

## 11. Implement artifact ownership

Updates require explicit ownership information.

Supported ownership classes:

### `inherited`

Provided dynamically by the platform. Repository normally contains no copied implementation.

### `managed`

Platform/profile owns the artifact and may update it.

### `merged`

Platform supplies part of the content while repository-specific content is preserved.

### `local`

Repository owns the artifact.

The update engine must not overwrite local artifacts.

Add tests for every ownership mode.

---

## 12. Implement `update` dry-run first

Do not begin with destructive updates.

Initial interface:

```bash
ai-native update --dry-run
```

The dry run should report:

```text
Current platform
Target compatible platform
Current profile
Target profile
Required migration chain
Files affected
Ownership of affected files
Files preserved
Warnings
Validation that will run afterwards
```

Example:

```text
Current platform: 1.0
Target platform: 1.2

Migrations:
1.0 → 1.1
1.1 → 1.2

Will modify:
.ai-native/manifest.yaml
.github/workflows/validate.yml

Will preserve:
AGENTS.md
README.md
.ai-native/local/*

Post-update validation:
ai-native doctor
pytest
pyright
pydocstyle
```

Definition of done:

- [ ] Dry-run makes no filesystem changes.
- [ ] Results are deterministic.
- [ ] Ownership is shown.
- [ ] Local files are identified as preserved.
- [ ] Migration order is correct.

---

## 13. Add migration infrastructure

Platform contract changes must be versioned.

Suggested structure:

```text
ai_native/
└── migrations/
    ├── v1_0_to_v1_1.py
    ├── v1_1_to_v1_2.py
    └── ...
```

Migration contract:

```python
can_apply()
plan()
apply()
validate()
```

Rules:

- migrations are sequential;
- migrations are deterministic;
- migrations are idempotent;
- migrations preserve local ownership;
- every migration has tests;
- updates stop safely on migration failure.

Critical invariant:

```text
update(update(repo)) == update(repo)
```

Running update twice against an already-current repository must produce no additional diff.

---

## 14. Implement actual `update`

Once dry-run and migrations are reliable:

```text
inspect current state
        ↓
resolve target compatible version
        ↓
calculate migration chain
        ↓
plan changes
        ↓
apply migrations
        ↓
preserve repository-owned content
        ↓
run doctor
        ↓
run repository validation
        ↓
produce summary
```

Update must fail safely.

If post-update validation fails:

- clearly report the failure;
- preserve evidence;
- avoid silently marking the update successful.

---

## 15. Create the `python-library` profile

Do not put Python assumptions into the platform core.

The initial profile should understand:

```text
pyproject.toml
pytest
pyright
pydocstyle
package builds
dependency metadata
wheel/sdist validation
PyPI metadata
semantic versioning
release validation
```

Profile relationship:

```text
AI Native Platform
        ↓
python-library
        ↓
Derived Python repository
```

Defaults must be extensible.

Example:

```yaml
validation:
  inherit: true
  append:
    - pytest tests/integration
```

---

## 16. Convert the existing AI-native checklist

Split the current checklist into three groups.

### Machine-checkable

Examples:

- manifest exists;
- inherited skills resolve;
- CI exists;
- deterministic validation is configured;
- release path exists;
- platform compatibility is valid.

These checks should migrate into `doctor`.

### Human-reviewable

Examples:

- API ergonomics;
- agent experience;
- documentation quality;
- product discoverability;
- permissions design.

These remain part of broader AI-native assessment.

### Advisory

Best practices that are useful but not compliance requirements.

The AI-native checklist becomes the **platform compliance specification**, not a separate parallel system.

---

## 17. Test strategy

The inheritance framework requires strong deterministic tests.

### Manifest

- [ ] Parse valid manifests.
- [ ] Reject invalid manifests.
- [ ] Handle schema versions.
- [ ] Validate compatibility ranges.

### Resolution

- [ ] Platform defaults.
- [ ] Profile inheritance.
- [ ] Repository extension.
- [ ] Append ordering.
- [ ] Overrides.
- [ ] Disable behavior.
- [ ] Conflict handling.

### Skills

- [ ] `welcome` composition.
- [ ] `troubleshooting` composition.
- [ ] Repository extension composition.

### Doctor

- [ ] Clean compliant repository passes.
- [ ] Missing manifest fails.
- [ ] Invalid profile fails.
- [ ] Unknown override warns/fails appropriately.
- [ ] Platform drift is detected.

### Update

- [ ] Dry-run changes nothing.
- [ ] Correct migration chain.
- [ ] Local files survive.
- [ ] Managed files update.
- [ ] Merged files preserve local content.
- [ ] Update is idempotent.
- [ ] Validation executes after update.
- [ ] Failed validation is reported correctly.

---

## 18. Dogfood on AI Native Platform

Before migrating another repository, the platform repository itself must use the new machinery where applicable.

Required:

- [ ] Platform manifest exists.
- [ ] `doctor` works on the platform repo.
- [ ] `welcome` understands the platform repo.
- [ ] `troubleshooting` works.
- [ ] Update development mode works.
- [ ] All framework validation passes.

Run:

```bash
ai-native doctor
```

Expected result:

```text
Status: compliant
```

Do not begin derived-repository migration before this succeeds.

---

## 19. PR strategy

Do not implement and migrate everything in one PR.

### PR 1 — Inheritance contract

Suggested title:

```text
feat: introduce AI-native inheritance contract
```

Scope:

- manifest schema;
- manifest parser;
- platform versioning;
- inheritance resolver;
- extension semantics;
- capability abstraction;
- `welcome`;
- `troubleshooting`;
- `doctor`;
- update dry-run skeleton;
- `python-library` profile skeleton;
- tests;
- documentation.

Explicitly exclude:

- derived repository migrations;
- broad platform redesign unrelated to inheritance;
- speculative profiles;
- large new skill catalog.

#### Merge gate

- [ ] Tests green.
- [ ] Static analysis green.
- [ ] Schema tests green.
- [ ] Inheritance tests green.
- [ ] Doctor tests green.
- [ ] Dry-run tests green.
- [ ] Documentation complete.
- [ ] CI green on current HEAD.
- [ ] Manual diff review complete.
- [ ] Codex review requested only when merge-ready.
- [ ] Useful Codex findings resolved.

Merge PR 1 before continuing.

### PR 2 — Update and migration engine

Scope:

- ownership model;
- migration infrastructure;
- update planning;
- update application;
- post-update validation;
- rollback/failure behavior where appropriate;
- idempotency tests.

Merge gate:

```text
update(update(repo)) == update(repo)
```

must hold.

### PR 3 — Platform dogfooding

Scope:

- apply the inheritance model to AI Native Platform itself;
- remove redundant platform-local duplication;
- prove `doctor`;
- prove `welcome`;
- prove `troubleshooting`;
- exercise update behavior.

Do not include external derived repositories.

### PR 4 — First derived repository pilot

Recommended first pilot:

```text
PermutiveAPI
```

Migration sequence:

```text
add manifest
        ↓
select python-library profile
        ↓
inherit welcome
        ↓
inherit troubleshooting
        ↓
inherits update
        ↓
remove duplicated platform behavior
        ↓
retain Permutive-specific extensions
        ↓
doctor
        ↓
existing repository validation
```

Any generic problem discovered during the pilot should be fixed in AI Native Platform rather than patched locally.

---

## 20. Prove inheritance actually propagates

Before declaring v1 successful, perform an intentional platform update.

Example:

```text
Platform 1.0
    ↓
change inherited welcome behavior
    ↓
Platform 1.1
```

Then on the pilot repository:

```bash
ai-native update --dry-run
```

Verify:

- [ ] Platform drift detected.
- [ ] Correct migration identified.
- [ ] Local extensions preserved.

Apply:

```bash
ai-native update
```

Then verify:

```bash
ai-native doctor
pytest
pyright
pydocstyle
```

Expected:

- [ ] Platform update propagated.
- [ ] No platform files had to be copied manually.
- [ ] Repository-specific customizations survived.
- [ ] Doctor passes.
- [ ] Existing repository validation passes.

If this does not work, the system is still templating rather than inheritance.

---

## 21. Migrate additional repositories

Only after the pilot proves the model.

Suggested migration order:

```text
simple Python libraries
        ↓
complex Python/data repositories
        ↓
services
        ↓
applications
        ↓
specialized platforms such as Android
```

Create new profiles only when a real repository demonstrates the need.

Do not pre-build a profile hierarchy speculatively.

---

## 22. CI integration

Derived repositories should eventually run:

```bash
ai-native doctor --ci
```

Recommended validation order:

```text
manifest validation
        ↓
ai-native doctor
        ↓
repository deterministic checks
```

Do not make AI review a deterministic CI dependency.

Codex review remains a semantic merge-ready review layer.

---

## 23. Versioning policy

Use semantic platform versions.

Example compatibility declaration:

```yaml
platform:
  compatible: "^1.0"
```

Within `1.x`, changes should preserve the v1 contract.

A major version may change:

- manifest semantics;
- inheritance rules;
- skill contracts;
- ownership semantics;
- profile behavior;
- migration guarantees.

Do not silently force derived repositories onto `latest`.

---

## 24. Self-improvement integration

Once inheritance is stable, reusable learnings should be promoted to the correct layer.

Classification:

```text
Repository-specific?
    → repository extension

Reusable by a repository family?
    → profile

Universal?
    → AI Native Platform
```

Knowledge flow:

```text
Derived repository
        ↑
      Profile
        ↑
AI Native Platform
```

Distribution flow:

```text
AI Native Platform improvement
        ↓
platform release
        ↓
update
        ↓
compatible derived repositories
```

This is the long-term mechanism for turning repository experience into platform improvement.

---

## 25. Explicit non-goals for v1

Do not expand the project unnecessarily.

V1 does **not** require:

- every possible repository profile;
- dozens of inherited skills;
- automatic cross-repository updates;
- autonomous merging;
- complex remote registries;
- dynamic plugin marketplaces;
- replacing existing deterministic tooling;
- rewriting repository-specific workflows that already work.

The purpose of v1 is to prove:

```text
inheritance
ownership
composition
validation
versioning
migration
update propagation
```

Everything else can build on those foundations.

---

## 26. Definition of done

AI Native inheritance v1 is complete only when the following works end to end:

```text
Compatible repository
        ↓
declares AI Native Platform
        ↓
selects a profile
        ↓
inherits welcome
        ↓
inherits troubleshooting
        ↓
inherits update
        ↓
doctor passes
        ↓
repository adds local extensions
        ↓
platform releases an improvement
        ↓
repository detects drift
        ↓
update --dry-run explains changes
        ↓
update applies migration
        ↓
local customizations survive
        ↓
doctor passes
        ↓
repository validation passes
```

Final acceptance checklist:

- [ ] No manual copying is required to receive inherited behavior.
- [ ] `welcome` is inherited.
- [ ] `troubleshooting` is inherited.
- [ ] `update` is inherited.
- [ ] `doctor` validates compliance.
- [ ] Profiles specialize platform behavior.
- [ ] Repository extensions compose cleanly.
- [ ] Ownership prevents accidental overwrites.
- [ ] Platform versions are explicit.
- [ ] Migrations are deterministic.
- [ ] Updates are idempotent.
- [ ] Dry-run accurately predicts updates.
- [ ] Platform improvements propagate to derived repositories.
- [ ] Deterministic CI remains the primary validation gate.
- [ ] Codex review is reserved for merge-ready semantic review.

---

## Execution rule

When implementation begins, optimize for the following order:

```text
contract
    ↓
resolution
    ↓
ownership
    ↓
doctor
    ↓
inherited skills
    ↓
update
    ↓
migration
    ↓
profile
    ↓
dogfood
    ↓
pilot
    ↓
rollout
```

Do not solve inheritance problems by adding repository-specific exceptions.

If the pilot exposes a generic weakness, fix the platform.

> The architecture is successful when "AI Native" is no longer something installed into each repository, but something repositories derive from.
