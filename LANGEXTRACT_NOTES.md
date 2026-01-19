# LangExtract investigation notes

## Goal

Identify how LangExtract could complement the existing response utilities and
validation helpers in `openai-sdk-helpers`.

## Initial questions

- Which LangExtract APIs are stable and intended for library integration?
- What inputs/outputs would map cleanly onto `openai_sdk_helpers.structure`?
- Are there overlap points with existing JSON schema validation utilities?
- Does LangExtract require specific model outputs or response formats?

## Next steps

- Review LangExtract usage examples and API surface.
- Prototype a small adapter around an existing helper function.
- Document a recommended integration path if it is a good fit.
