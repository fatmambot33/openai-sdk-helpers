from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from openai_sdk_helpers.response.base import ResponseBase
from openai_sdk_helpers.streamlit_app import StreamlitAppConfig, load_app_config
from openai_sdk_helpers.structure.base import BaseStructure


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


def test_vector_store_normalization_returns_copy(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
APP_CONFIG = {"build_response": lambda: None, "system_vector_store": "files"}
""",
    )

    config = StreamlitAppConfig.load_app_config(config_path=config_path)

    stores = config.normalized_vector_stores()
    stores.append("mutated")

    assert config.system_vector_store == ["files"]
    assert config.normalized_vector_stores() == ["files"]


def test_load_app_config_proxy(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
APP_CONFIG = {"build_response": lambda: None}
""",
    )

    config = load_app_config(config_path=config_path)

    assert isinstance(config, StreamlitAppConfig)


class _DummyResponse(ResponseBase[BaseStructure]):
    """Minimal :class:`ResponseBase` subclass for config construction tests.

    Methods
    -------
    __init__()
        Configure a stub response session for testing.
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="hi",
            tools=[],
            schema=None,
            output_structure=None,
            tool_handlers={},
            client=object(),
            model="dummy",
        )


def test_response_base_builds_streamlit_config() -> None:
    config = _DummyResponse.build_streamlit_config(
        display_title="Custom title",
        description="Custom description",
        system_vector_store="files",
        preserve_vector_stores=True,
        model="dummy",
    )

    assert isinstance(config, StreamlitAppConfig)
    assert config.display_title == "Custom title"
    assert config.description == "Custom description"
    assert config.preserve_vector_stores is True
    assert config.model == "dummy"
    assert config.system_vector_store == ["files"]

    response_instance = config.build_response()

    assert isinstance(response_instance, _DummyResponse)


def test_config_accepts_response_alias(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
from tests.test_streamlit_configuration import _DummyResponse

APP_CONFIG = {"response": _DummyResponse, "display_title": "Alias title"}
""",
    )

    config = StreamlitAppConfig.load_app_config(config_path=config_path)

    assert config.display_title == "Alias title"
    response_instance = config.build_response()

    assert isinstance(response_instance, ResponseBase)
    assert response_instance.__class__.__name__ == "_DummyResponse"


def test_config_accepts_response_class_directly(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
from tests.test_streamlit_configuration import _DummyResponse

APP_CONFIG = _DummyResponse
""",
    )

    config = StreamlitAppConfig.load_app_config(config_path=config_path)

    response_instance = config.build_response()

    assert isinstance(response_instance, ResponseBase)
    assert response_instance.__class__.__name__ == "_DummyResponse"
