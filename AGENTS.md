# AGENTS.md

These instructions apply to any automated agent contributing to this repository.

## 1. Code Style

- Use standard PEP 8 formatting for all Python code.
- Write commit messages in the imperative mood (e.g., "Add feature" not "Added feature").
- Keep the implementation Pythonic and maintainable.

## 2. Docstring Style

- Use **NumPy-style** docstrings following [PEP 257](https://peps.python.org/pep-0257/) conventions.
- **Do not** use `:param` / `:type` syntax (reST/Sphinx style) or Google-style `Args`.
- Always include type hints in function signatures.
- Begin docstrings with a **short, one-line summary**, followed by a blank line and an optional extended description.
- Use the following NumPy-style sections:
  - `Parameters`
  - `Returns`
  - `Raises`
  - `Examples`
- Document all public classes, methods, and functions.
- **For classes, the main docstring should include a `Methods` section summarizing each public method and its one-line description.**
- For optional parameters, note the default value in the description.
- Use imperative present tense and active voice (“Return…”, “Fetch…”).

## 3. Code Quality and Testing

Before running tests, install the development dependencies:

```bash
# Install the package with dev dependency group (PEP 735)
pip install -e . --group dev
```

To ensure your changes will pass the automated checks in our Continuous Integration (CI) pipeline, run the following commands locally before committing. All checks must pass.

**Style Checks:**
```bash
pydocstyle src
black --check .
```

**Static Type Analysis:**
```bash
pyright src
```

**Unit Tests and Coverage:**
```bash
pytest -q --cov=src --cov-report=term-missing --cov-fail-under=70
```

## 4. Directory Layout

- Production code lives in `src/`.
- Tests live in `tests/`.
- Keep imports relative within the package (e.g., `from rda_bundle...`).

## 5. Pull Request Messages

Each pull request should include:

1. **Summary** – brief description of the change.
2. **Testing** – commands run and confirmation that the tests passed.

Example PR body:

```
### Summary
- add new helper to utils.list
- expand tests for list chunking

### Testing
- `pytest` (all tests passed)
```

## 6. General Guidelines

- Avoid pushing large data files to the repository.
- Prefer small, focused commits over sweeping changes.
- Update or add tests whenever you modify functionality.
