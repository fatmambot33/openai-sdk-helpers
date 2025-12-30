from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from openai_sdk_helpers.streamlit_app import StreamlitAppConfig, load_app_config


def _write_config(tmp_path: Path, body: str) -> Path:
    config_path = tmp_path / "temp_config.py"
    config_path.write_text(body)
    return config_path


def test_load_app_config_success(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
APP_CONFIG = {"build_response": lambda: None}
""",
    )

    config = StreamlitAppConfig.load_app_config(config_path=config_path)

    assert config.display_title == "Example copilot"
    assert config.normalized_vector_stores() == []


def test_missing_config_file() -> None:
    missing = Path("/tmp/does/not/exist.py")
    with pytest.raises(FileNotFoundError):
        StreamlitAppConfig.load_app_config(config_path=missing)


def test_missing_app_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "build_response = lambda: None\n")

    with pytest.raises(ValueError):
        StreamlitAppConfig.load_app_config(config_path=config_path)


def test_invalid_builder(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
APP_CONFIG = {"build_response": "not_callable"}
""",
    )

    with pytest.raises(ValidationError):
        StreamlitAppConfig.load_app_config(config_path=config_path)


def test_invalid_vector_store_type(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
APP_CONFIG = {"build_response": lambda: None, "system_vector_store": [1, 2]}
""",
    )

    with pytest.raises(ValidationError):
        StreamlitAppConfig.load_app_config(config_path=config_path)


def test_load_app_config_proxy(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
APP_CONFIG = {"build_response": lambda: None}
""",
    )

    config = load_app_config(config_path=config_path)

    assert isinstance(config, StreamlitAppConfig)
