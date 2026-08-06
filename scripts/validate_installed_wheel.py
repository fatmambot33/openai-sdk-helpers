"""Validate runtime contents from an installed package wheel."""

from __future__ import annotations

import importlib
import os
from importlib import resources
from importlib.metadata import distribution
from pathlib import Path

_DISTRIBUTION_NAME = "openai-sdk-helpers"
_PACKAGE_NAME = "openai_sdk_helpers"
_REQUIRED_CONSOLE_SCRIPTS = {
    "openai-helpers",
    "openai-helpers-credentials",
}
_REQUIRED_PACKAGE_FILES = {
    "openai_sdk_helpers/py.typed",
    "openai_sdk_helpers/prompt/summarizer.jinja",
    "openai_sdk_helpers/prompt/translator.jinja",
    "openai_sdk_helpers/prompt/validator.jinja",
    "openai_sdk_helpers/prompt/vector_planner.jinja",
}


def main() -> int:
    """Validate the installed distribution and its runtime package data.

    Returns
    -------
    int
        Zero when the installed wheel contains every required runtime surface.
    """
    if os.environ.get("OPENAI_API_KEY"):
        raise AssertionError("Installed-wheel validation must run without credentials")

    package = importlib.import_module(_PACKAGE_NAME)
    package_path = Path(package.__file__ or "").resolve()
    if not package_path.is_file():
        raise AssertionError("Installed package does not have a resolvable module path")

    installed_distribution = distribution(_DISTRIBUTION_NAME)
    installed_files = {
        str(path).replace("\\", "/") for path in (installed_distribution.files or ())
    }
    missing_files = sorted(_REQUIRED_PACKAGE_FILES - installed_files)
    if missing_files:
        raise AssertionError(f"Wheel is missing runtime files: {missing_files}")

    console_scripts = {
        entry_point.name
        for entry_point in installed_distribution.entry_points
        if entry_point.group == "console_scripts"
    }
    missing_scripts = sorted(_REQUIRED_CONSOLE_SCRIPTS - console_scripts)
    if missing_scripts:
        raise AssertionError(f"Wheel is missing console scripts: {missing_scripts}")

    package_resources = resources.files(_PACKAGE_NAME)
    if not package_resources.joinpath("py.typed").is_file():
        raise AssertionError("py.typed is not accessible through importlib.resources")

    prompt_resources = resources.files(f"{_PACKAGE_NAME}.prompt")
    for prompt_name in (
        "summarizer.jinja",
        "translator.jinja",
        "validator.jinja",
        "vector_planner.jinja",
    ):
        if not prompt_resources.joinpath(prompt_name).is_file():
            raise AssertionError(f"Prompt template is unavailable: {prompt_name}")

    print(f"Installed wheel validated at {package_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
