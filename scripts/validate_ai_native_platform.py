"""Validate declarations and concrete AI-native repository evidence."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(".")
MANIFEST = ROOT / "AI_NATIVE_PLATFORM.yaml"
TRUE_PATHS = (
    ("product", "ai_native"), ("product", "plugin_first"), ("product", "sdk_first"),
    ("plugin", "enabled"), ("plugin", "codex", "supported"), ("plugin", "codex", "marketplace"),
    ("plugin", "discovery", "entry_points"), ("plugin", "discovery", "manifest"), ("plugin", "discovery", "capabilities"),
    ("plugin", "credentials", "local_only"), ("plugin", "credentials", "policy", "never_store_remote"),
    ("plugin", "credentials", "policy", "never_commit"), ("plugin", "credentials", "policy", "never_echo"),
    ("interfaces", "sdk"), ("interfaces", "cli"), ("interfaces", "plugin"), ("interfaces", "json_schema"),
    ("quality", "typed"), ("quality", "tests"), ("quality", "docs"), ("quality", "examples"),
    ("quality", "security_scan"), ("self_improvement", "enabled"), ("self_improvement", "github", "issues"),
    ("self_improvement", "autonomous", "discover_improvements"), ("self_improvement", "autonomous", "create_issues"),
    ("self_improvement", "autonomous", "generate_pr"), ("self_improvement", "autonomous", "run_ci"),
    ("release", "block_if_quality_fails"), ("release", "block_if_plugin_invalid"),
)
COMMANDS = {"validate", "test", "docs", "examples", "upgrade", "uninstall"}
GUARANTEES = {"deterministic_tool_discovery", "structured_outputs", "issue_driven_improvement", "ci_validated_changes", "governed_autonomy"}
APPROVALS = {"breaking_changes", "security_changes", "credential_changes", "public_api_changes", "permission_expansion", "release_changes"}

def get(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(path))
        current = current[key]
    return current

def matches(*patterns: str) -> list[Path]:
    return sorted({p for pattern in patterns for p in ROOT.glob(pattern) if p.is_file() and ".git" not in p.parts})

def contents(paths: list[Path]) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in paths if p.stat().st_size < 2_000_000).lower()

def evidence(data: dict[str, Any]) -> list[str]:
    docs = contents(matches("README.md", "docs/**/*.md"))
    code = contents(matches("src/**/*.py", "**/plugins/**/*.py", "plugins/**/*.py"))
    ci = contents(matches(".github/workflows/*.yml", ".github/workflows/*.yaml"))
    tests = matches("tests/test_*.py", "tests/**/*test*.py")
    pyproject = ROOT / "pyproject.toml"
    checks = {
        "pyproject.toml": pyproject.is_file(),
        "Codex plugin manifest": bool(matches(".codex-plugin/plugin.json", "plugins/**/.codex-plugin/plugin.json")),
        "Codex marketplace catalog": bool(matches(".agents/plugins/marketplace.json", "plugins/**/marketplace.json")),
        "typed plugin contract": "plugin" in code and ("protocol" in code or "abstractbaseclass" in code),
        "typing marker or Pyright contract": bool(matches("src/**/py.typed", "**/py.typed")) or (pyproject.is_file() and "pyright" in pyproject.read_text(encoding="utf-8", errors="ignore").lower()),
        "strict type checking in CI": "pyright" in ci or "mypy" in ci,
        "plugin tests": any("plugin" in p.name.lower() for p in tests),
        "general tests": bool(tests),
        "AGENTS.md": (ROOT / "AGENTS.md").is_file(),
        "PyPI installation documentation": "pip install" in docs,
        "Git installation documentation": "git+https://" in docs or "git clone" in docs,
        "editable installation documentation": "pip install -e" in docs,
        "plugin documentation": "plugin" in docs and "codex" in docs,
        "AI improvement issue template": (ROOT / ".github/ISSUE_TEMPLATE/ai-improvement.yml").is_file(),
        "self-improvement workflow": (ROOT / ".github/workflows/ai-self-improvement.yml").is_file() or (ROOT / ".github/workflows/ai-self-improve.yml").is_file(),
    }
    credentials = data.get("plugin", {}).get("credentials", {})
    if credentials.get("required"):
        checks["credential template"] = isinstance(credentials.get("env_example"), str) and (ROOT / credentials["env_example"]).is_file()
        checks["configure command"] = bool(credentials.get("setup_command"))
        checks["doctor command"] = bool(credentials.get("validation_command"))
        checks[".env ignored"] = (ROOT / ".gitignore").is_file() and ".env" in (ROOT / ".gitignore").read_text(encoding="utf-8", errors="ignore")
    return [name for name, ok in checks.items() if not ok]

def main() -> int:
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else None
    if not isinstance(data, dict) or data.get("version") != 1:
        print("AI-native platform validation failed:\n- missing or invalid version 1 manifest")
        return 1
    errors: list[str] = []
    standard = data.get("standard", {})
    if standard.get("repository") != "fatmambot33/ai-native-platform": errors.append("invalid standard.repository")
    if not re.fullmatch(r"[0-9a-f]{40}|v?\d+\.\d+\.\d+", str(standard.get("ref", ""))): errors.append("standard.ref must be immutable")
    for path in TRUE_PATHS:
        try:
            if get(data, path) is not True: errors.append(f"{'.'.join(path)} must be true")
        except KeyError: errors.append(f"missing {'.'.join(path)}")
    declared = set(data.get("commands", {}).get("required", []))
    errors.extend(f"commands.required must include {x}" for x in sorted(COMMANDS - declared))
    approvals = data.get("self_improvement", {}).get("governance", {}).get("human_approval", {})
    missing = sorted(x for x in APPROVALS if approvals.get(x) is not True)
    if missing: errors.append("missing human approval gates: " + ", ".join(missing))
    missing = sorted(GUARANTEES - set(data.get("agent", {}).get("guarantees", [])))
    if missing: errors.append("missing agent guarantees: " + ", ".join(missing))
    errors.extend(f"missing repository evidence: {x}" for x in evidence(data))
    if errors:
        print("AI-native platform validation failed:")
        for error in errors: print(f"- {error}")
        return 1
    print("AI-native platform validation passed with repository evidence.")
    return 0

if __name__ == "__main__": raise SystemExit(main())
