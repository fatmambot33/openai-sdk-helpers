# Decisions API

`openai_sdk_helpers.decisions` provides a thin, capability-gated helper surface
for the OpenAI Decisions API.

Use Decisions when an application needs a fast typed judgment over shared input
without asking a generative response or agent loop to perform the same routing,
classification, prioritization, or scoring step.

## Compatibility

The package-wide OpenAI Python SDK range remains:

```text
openai>=2.45.0,<4.0.0
```

The Decisions API helper surface requires an SDK client exposing
`client.decisions.create`, first available in `openai>=3.26.0`.

Applications that need Decisions can upgrade only that dependency:

```bash
pip install "openai>=3.26.0,<4.0.0"
```

Importing `openai_sdk_helpers.decisions` remains safe on older package-supported
SDK versions. Constructing a Decisions helper with an incompatible client raises
`DecisionsUnavailableError` before any network request.

## Choosing Decisions versus other surfaces

Use **Decisions** for typed predicate probabilities, fixed-choice selection, and
ordered scoring over shared text or supported inline-image input.

Use **Responses** when the application needs generated text, structured output,
tool calls, richer reasoning workflows, conversation continuation, or other
Responses API features.

Use **Agents** when the application needs an application-run agent loop,
handoffs, sessions, guardrails, tools, or tracing.

Use **Managed Agents** when OpenAI-managed durable agent sessions and execution
are required.

The Decisions helpers deliberately do not become a routing framework, policy
engine, explanation layer, or replacement schema.

## Synchronous usage

```python
from openai import OpenAI
from openai_sdk_helpers.decisions import DecisionsClient

client = DecisionsClient(OpenAI())

answer = client.predicate(
    "This request is a normal account lookup.",
    instructions="Should this request use the standard support path?",
    name="standard_path",
)

if answer.type == "predicate":
    print(answer.probability)
elif answer.type == "refusal":
    print("The model refused this question.")
```

The convenience method returns the original SDK answer object. A predicate,
choice, or score request can therefore still return a native refusal.

## Choice decisions

```python
answer = client.choose(
    "Customer asks for an invoice copy.",
    instructions="Choose the best support queue.",
    name="queue",
    choices=(
        {"value": "billing", "description": "Invoices and payment questions"},
        {"value": "account", "description": "Account settings and access"},
        {"value": "technical", "description": "Product or API failures"},
    ),
)

if answer.type == "choice":
    print(answer.choice, answer.confidence)
```

Choice values preserve the official API's typed string-or-boolean semantics.

## Score decisions

```python
answer = client.score(
    "The proposed answer solves the issue and cites the relevant policy.",
    instructions="Score answer quality.",
    name="quality",
    levels=(
        {"label": "poor", "description": "Does not solve the issue"},
        {"label": "good", "description": "Solves the issue adequately"},
        {"label": "excellent", "description": "Clear, complete, and precise"},
    ),
)

if answer.type == "score":
    print(answer.score, answer.confidence)
```

Levels are ordered as supplied to the official SDK.

## Batch questions

Use `create` when several questions should be evaluated against the same input:

```python
decision = client.create(
    input="Candidate request",
    questions=(
        {
            "type": "predicate",
            "name": "eligible",
            "instructions": "Is this request eligible for automation?",
        },
        {
            "type": "choice",
            "name": "route",
            "instructions": "Choose the destination.",
            "choices": (
                {"value": "fast_path"},
                {"value": "review"},
                {"value": "reject"},
            ),
        },
    ),
)

for answer in decision.answers:
    print(answer)
```

The original `Decision` object is returned, including model and usage data.

## Asynchronous usage

```python
from openai import AsyncOpenAI
from openai_sdk_helpers.decisions import AsyncDecisionsClient

client = AsyncDecisionsClient(AsyncOpenAI())

answer = await client.predicate(
    "A standard lookup request",
    instructions="Should this use the standard path?",
)
```

The asynchronous helper mirrors the synchronous surface and preserves the
official asynchronous resource and result objects.

## Capability detection

Applications supporting multiple OpenAI SDK generations can detect the surface
before construction:

```python
from openai_sdk_helpers.decisions import decisions_available

if decisions_available(client):
    ...
```

The structural check verifies that `client.decisions.create` is callable.

## Raw SDK escape hatch

Both facades expose:

- `sdk_client` — the configured official OpenAI client;
- `decisions` — the raw official Decisions resource.

Use the raw resource whenever a newly released parameter or behavior has not yet
earned a package-level helper. The package does not intentionally hide upstream
SDK capabilities.

## Model behavior

The helper defaults to `gpt-6-luna`, the model supported by the Decisions API
at launch. The `model` parameter remains overridable so applications are not
locked to that launch-time model when OpenAI expands support.

## Observability

Every network helper accepts an optional `OperationContext`. The context wraps
the request lifecycle without normalizing or replacing the official SDK result
or exception.

## Input boundary

The helper forwards Decisions input unchanged. The official API currently
supports a string or user messages containing text and inline images. Callers
should use the official SDK types for multimodal message construction so new
upstream fields remain visible instead of being hidden behind a second package
schema.
