# Examples

Examples are classified by the compatibility promise they carry.

## Supported smoke examples

Supported examples are deterministic, credential-free, and executed from the
built wheel in CI. A public API change must update these examples in the same
pull request.

| Example | Surface | Network behavior |
| --- | --- | --- |
| `supported_responses.py` | Responses configuration and registry | No API call |
| `supported_agents.py` | Agents configuration and official SDK agent construction | No API call |
| `codex_plugin.py` | Codex plugin lifecycle plus sync and async commands | No API call |

Run them after installing the package:

```bash
python examples/supported_responses.py
python examples/supported_agents.py
python examples/codex_plugin.py
```

## Illustrative examples

The remaining examples demonstrate broader workflows or integration patterns.
They are reviewed for usefulness, but they are not part of the executable
compatibility contract and may require credentials, files, optional extras, or
external API calls:

- `automatic_file_detection.py`
- `classify.py`
- `image_and_file_support.py`
- `registry_features_demo.py`
- `textextract.py`
- `tool_spec_usage_example.py`
- `tool_wrapper_and_async_demo.py`

Illustrative examples must state their prerequisites in their module docstring
when they require credentials, external files, or an optional installation
profile.

## Deprecated examples

No example is currently designated deprecated. When an example no longer
represents a supported workflow, mark it deprecated here, add its replacement,
and remove it only under the package compatibility policy.

## Policy

A supported example must:

- run from an installed wheel rather than an editable source install;
- avoid `OPENAI_API_KEY` and external network access;
- use intentional public imports;
- assert its expected behavior instead of only printing output;
- complete on every pull request in the package smoke workflow.

Examples that need live OpenAI behavior remain illustrative and belong in a
separate opt-in integration environment, not pull-request CI.
