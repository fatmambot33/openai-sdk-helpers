"""Local-only credential setup for openai-sdk-helpers."""

from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

from dotenv import dotenv_values

REQUIRED_VARIABLES = ("OPENAI_API_KEY",)
OPTIONAL_VARIABLES = ("OPENAI_PROJECT", "OPENAI_ORG_ID")


def _is_ignored(env_path: Path) -> bool:
    """Return whether a nearby gitignore explicitly ignores ``.env``."""
    for parent in (env_path.parent, *env_path.parents):
        gitignore = parent / ".gitignore"
        if gitignore.is_file():
            entries = {
                line.strip()
                for line in gitignore.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            }
            return ".env" in entries or "*.env" in entries
        if (parent / ".git").exists():
            break
    return False


def _write_env(path: Path, values: Dict[str, str]) -> None:
    """Write local credentials with restrictive permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{name}={value}\n" for name, value in values.items() if value),
        encoding="utf-8",
    )
    try:
        path.chmod(0o600)
    except OSError:
        pass


def configure(env_file: Path, *, force: bool = False) -> int:
    """Interactively create a local OpenAI credential file."""
    if env_file.exists() and not force:
        print(f"Refusing to overwrite {env_file}. Use --force to replace it.")
        return 2

    print("OpenAI SDK Helpers local credential setup")
    print("Credentials remain in this local .env file and are never uploaded.")
    values = {
        "OPENAI_API_KEY": getpass.getpass("OpenAI API key: ").strip(),
        "OPENAI_PROJECT": input("OpenAI project ID (optional): ").strip(),
        "OPENAI_ORG_ID": input("OpenAI organization ID (optional): ").strip(),
    }
    if not values["OPENAI_API_KEY"]:
        print("Missing required value: OPENAI_API_KEY")
        return 2

    _write_env(env_file, values)
    print(f"Created {env_file} with local-only permissions where supported.")
    if not _is_ignored(env_file):
        print("WARNING: add .env to the repository .gitignore before committing.")
        return 1
    print("Credential setup complete. Run `openai-helpers-credentials doctor`.")
    return 0


def doctor(env_file: Path) -> int:
    """Validate local OpenAI credentials without displaying values."""
    problems = []
    if not env_file.is_file():
        problems.append(f"missing credential file: {env_file}")
        values: Dict[str, Optional[str]] = {}
    else:
        values = dict(dotenv_values(env_file))
        missing = [name for name in REQUIRED_VARIABLES if not values.get(name)]
        if missing:
            problems.append("missing variables: " + ", ".join(missing))
        if os.name != "nt" and env_file.stat().st_mode & 0o077:
            problems.append("credential file permissions are too broad; run chmod 600 .env")
    if not _is_ignored(env_file):
        problems.append(".env is not explicitly ignored by .gitignore")

    if problems:
        print("OpenAI SDK Helpers credential check failed:")
        for problem in problems:
            print(f"- {problem}")
        print("Run `openai-helpers-credentials configure` to repair it.")
        return 1

    configured_optional = [name for name in OPTIONAL_VARIABLES if values.get(name)]
    print("OpenAI SDK Helpers local credential check passed.")
    print("OPENAI_API_KEY is present; its value was not displayed.")
    if configured_optional:
        print("Optional variables present: " + ", ".join(configured_optional))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the credential command parser."""
    parser = argparse.ArgumentParser(prog="openai-helpers-credentials")
    subparsers = parser.add_subparsers(dest="command", required=True)
    configure_parser = subparsers.add_parser("configure")
    configure_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    configure_parser.add_argument("--force", action="store_true")
    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--env-file", type=Path, default=Path(".env"))
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the local credential command."""
    args = build_parser().parse_args(argv)
    if args.command == "configure":
        return configure(args.env_file, force=args.force)
    return doctor(args.env_file)


if __name__ == "__main__":
    raise SystemExit(main())
