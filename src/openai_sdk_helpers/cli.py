"""Command-line interface for openai-sdk-helpers development.

Provides CLI commands for testing agents, validating templates, inspecting
response configurations, and inspecting installed Codex plugins.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path


def cmd_agent_test(args: argparse.Namespace) -> int:
    """Test an agent locally.

    Parameters
    ----------
    args
        Command arguments containing ``agent_name`` and ``input``.

    Returns
    -------
    int
        Exit code.
    """
    print(f"Testing agent: {args.agent_name}")
    print(f"Input: {args.input}")
    print("\n[Not yet implemented - agent testing framework coming soon]")
    return 0


def cmd_template_validate(args: argparse.Namespace) -> int:
    """Validate Jinja2 templates.

    Parameters
    ----------
    args
        Command arguments containing ``template_path``.

    Returns
    -------
    int
        Zero for success and one for validation errors.
    """
    from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError

    template_path = Path(args.template_path)
    if not template_path.exists():
        print(f"Error: Path not found: {template_path}", file=sys.stderr)
        return 1

    if template_path.is_file():
        templates = [template_path]
        base_dir = template_path.parent
    else:
        templates = list(template_path.glob("**/*.jinja"))
        base_dir = template_path

    if not templates:
        print(f"No .jinja templates found in {template_path}")
        return 0

    env = Environment(loader=FileSystemLoader(base_dir))
    errors = []
    for template_file in templates:
        relative_path = template_file.relative_to(base_dir)
        try:
            env.get_template(str(relative_path))
            print(f"✓ {relative_path}")
        except TemplateSyntaxError as exc:
            errors.append((relative_path, str(exc)))
            print(f"✗ {relative_path}: {exc}", file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} template(s) with errors", file=sys.stderr)
        return 1

    print(f"\n{len(templates)} template(s) validated successfully")
    return 0


def cmd_registry_list(args: argparse.Namespace) -> int:
    """List all registered response configurations.

    Parameters
    ----------
    args
        Parsed command arguments.

    Returns
    -------
    int
        Exit code.
    """
    del args
    try:
        from openai_sdk_helpers import get_default_registry
    except ImportError:
        print("Error: openai_sdk_helpers not installed", file=sys.stderr)
        return 1

    registry = get_default_registry()
    names = registry.list_names()
    if not names:
        print("No configurations registered")
        return 0

    print("Registered configurations:")
    for name in sorted(names):
        configuration = registry.get(name)
        tools_count = len(configuration.tools) if configuration.tools else 0
        print(f"  - {name} ({tools_count} tools)")
    return 0


def cmd_registry_inspect(args: argparse.Namespace) -> int:
    """Inspect a specific response configuration.

    Parameters
    ----------
    args
        Command arguments containing ``config_name``.

    Returns
    -------
    int
        Zero for success and one when the configuration is absent.
    """
    try:
        from openai_sdk_helpers import get_default_registry
    except ImportError:
        print("Error: openai_sdk_helpers not installed", file=sys.stderr)
        return 1

    registry = get_default_registry()
    try:
        configuration = registry.get(args.config_name)
    except KeyError:
        print(f"Error: Configuration '{args.config_name}' not found", file=sys.stderr)
        print("\nAvailable configurations:")
        for name in sorted(registry.list_names()):
            print(f"  - {name}")
        return 1

    print(f"Configuration: {configuration.name}")
    instructions_str = str(configuration.instructions)
    instructions_preview = (
        instructions_str[:100] if len(instructions_str) > 100 else instructions_str
    )
    print(f"Instructions: {instructions_preview}...")
    print(f"Tools: {len(configuration.tools) if configuration.tools else 0}")
    if configuration.tools:
        print("\nTool names:")
        for tool in configuration.tools:
            tool_name = "unknown"
            if isinstance(tool, Mapping):
                function_value = tool.get("function")
                if isinstance(function_value, Mapping):
                    name_value = function_value.get("name")
                    if isinstance(name_value, str) and name_value:
                        tool_name = name_value
            print(f"  - {tool_name}")
    return 0


def cmd_codex_plugins(args: argparse.Namespace) -> int:
    """Discover and list installed Codex plugins.

    Parameters
    ----------
    args
        Parsed command arguments.

    Returns
    -------
    int
        Zero when discovery succeeds and one when any plugin fails to load.
    """
    del args
    from openai_sdk_helpers.codex import CodexPluginRegistry

    registry = CodexPluginRegistry()
    report = registry.discover_isolated()
    inspections = registry.inspect_plugins()
    if not inspections:
        print("No Codex plugins discovered")
    else:
        print("Codex plugins:")
        for inspection in inspections:
            metadata = inspection.metadata
            capabilities = ", ".join(metadata.capabilities) or "none"
            deprecated = " [deprecated]" if metadata.deprecated else ""
            print(f"  - {metadata.name} {metadata.version}{deprecated}")
            print(f"    capabilities: {capabilities}")
            if metadata.summary:
                print(f"    {metadata.summary}")

    _print_codex_discovery_failures(report.failures)
    return 0 if report.ok else 1


def cmd_codex_commands(args: argparse.Namespace) -> int:
    """Discover installed Codex plugins and list their commands.

    Parameters
    ----------
    args
        Parsed command arguments.

    Returns
    -------
    int
        Zero when discovery succeeds and one when any plugin fails to load.
    """
    del args
    from openai_sdk_helpers.codex import CodexPluginRegistry

    registry = CodexPluginRegistry()
    report = registry.discover_isolated()
    inspections = registry.inspect_plugins()
    commands_found = False
    for inspection in inspections:
        for command_name in inspection.command_names:
            if not commands_found:
                print("Codex commands:")
                commands_found = True
            print(f"  - {command_name} ({inspection.metadata.name})")
    if not commands_found:
        print("No Codex commands discovered")

    _print_codex_discovery_failures(report.failures)
    return 0 if report.ok else 1


def _print_codex_discovery_failures(failures: tuple[object, ...]) -> None:
    if not failures:
        return
    print("Codex plugin discovery failures:", file=sys.stderr)
    for failure in failures:
        entry_point = getattr(failure, "entry_point", "unknown")
        error_type = getattr(failure, "error_type", "Error")
        message = getattr(failure, "message", "")
        print(
            f"  - {entry_point}: {error_type}: {message}",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI interface.

    Parameters
    ----------
    argv
        Command-line arguments. Uses ``sys.argv`` when omitted.

    Returns
    -------
    int
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="openai-helpers",
        description="OpenAI SDK Helpers CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    agent_parser = subparsers.add_parser("agent", help="Agent operations")
    agent_sub = agent_parser.add_subparsers(dest="agent_command")
    test_parser = agent_sub.add_parser("test", help="Test an agent")
    test_parser.add_argument("agent_name", help="Agent name to test")
    test_parser.add_argument("--input", default="", help="Test input")

    template_parser = subparsers.add_parser("template", help="Template operations")
    template_sub = template_parser.add_subparsers(dest="template_command")
    validate_parser = template_sub.add_parser("validate", help="Validate templates")
    validate_parser.add_argument(
        "template_path",
        help="Path to template file or directory",
    )

    registry_parser = subparsers.add_parser("registry", help="Registry operations")
    registry_sub = registry_parser.add_subparsers(dest="registry_command")
    registry_sub.add_parser("list", help="List registered configurations")
    inspect_parser = registry_sub.add_parser("inspect", help="Inspect configuration")
    inspect_parser.add_argument("config_name", help="Configuration name")

    codex_parser = subparsers.add_parser("codex", help="Codex plugin operations")
    codex_sub = codex_parser.add_subparsers(dest="codex_command")
    codex_sub.add_parser("plugins", help="List installed Codex plugins")
    codex_sub.add_parser("commands", help="List installed Codex commands")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    if args.command == "agent" and args.agent_command == "test":
        return cmd_agent_test(args)
    if args.command == "template" and args.template_command == "validate":
        return cmd_template_validate(args)
    if args.command == "registry":
        if args.registry_command == "list":
            return cmd_registry_list(args)
        if args.registry_command == "inspect":
            return cmd_registry_inspect(args)
    if args.command == "codex":
        if args.codex_command == "plugins":
            return cmd_codex_plugins(args)
        if args.codex_command == "commands":
            return cmd_codex_commands(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
