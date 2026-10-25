# SDK Compatibility Matrix

This document tracks current dependency requirements.

## Supported SDK versions

| openai-sdk-helpers | openai | openai-agents | Python | Status |
| --- | --- | --- | --- | --- |
| 0.11.1+ | >=3.37.0,<4 | >=0.24.0,<1 | 3.10–3.13 | Active |
| 0.11.0 and earlier | See each release's declared requirements | See release | See release | Historical |

## OpenAI Python SDK

The package-wide minimum is `openai>=3.37.0,<4.0.0` across **all**
installation profiles: Responses, Agents, Managed Agents, retrieval, and
Decisions. Older OpenAI SDK releases are not supported by 0.11.1+.

The minimum incorporates the 2026-10-22 Decisions GA release, including
JSON, text, images, conversation/tool history, structured question
instructions, prompt caching, and batches of up to 400 questions.

Injected SDK clients must still provide `client.decisions.create` or
`client.beta.agents.sessions` for their respective helpers. Missing
capabilities raise `DecisionsUnavailableError` or
`ManagedAgentsUnavailableError` before any network call. The helpers
preserve upstream SDK objects, exceptions, and raw escape hatches.

## OpenAI Agents SDK

The supported range is now `openai-agents>=0.24.0,<1.0.0`. The older
0.18.x releases constrain `openai` to `<3`, so they cannot be installed
with the package-wide `openai>=3.37.0` minimum. Applications can apply
narrower pins where reproducibility matters.

## Package requirements

```toml
dependencies = [
    "openai>=3.37.0,<4.0.0",
    "openai-agents>=0.24.0,<1.0.0",
]
```

## Testing

CI pins `openai==3.37.0` and `openai-agents==0.24.0` in its minimum
SDK job. Python 3.10–3.13, clean installation profiles, typing, tests,
and credential-free installed-wheel examples remain release gates.

Earlier releases remain immutable and retain their own dependency
requirements; consult their tags and changelog for historical support.
