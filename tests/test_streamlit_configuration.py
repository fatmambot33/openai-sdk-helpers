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


def test_load_app_config_success(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
from openai_sdk_helpers.response.base import ResponseBase
from openai_sdk_helpers.structure.base import BaseStructure


class TempResponse(ResponseBase[BaseStructure]):
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


APP_CONFIG = {"response": TempResponse}
""",
    )

    config = StreamlitAppConfig.load_app_config(config_path=config_path)

    assert config.display_title == "Example copilot"
    assert config.normalized_vector_stores() == []
    assert isinstance(config.create_response(), ResponseBase)


def test_missing_config_file() -> None:
    missing = Path("/tmp/does/not/exist.py")
    with pytest.raises(FileNotFoundError):
        StreamlitAppConfig.load_app_config(config_path=missing)


def test_missing_app_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, "build_response = lambda: None\n")

    with pytest.raises(ValueError):
        StreamlitAppConfig.load_app_config(config_path=config_path)


def test_missing_response(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
APP_CONFIG = {}
""",
    )

    with pytest.raises(ValidationError):
        StreamlitAppConfig.load_app_config(config_path=config_path)


def test_invalid_response_type(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
APP_CONFIG = {"response": "not_callable"}
""",
    )

    with pytest.raises(ValidationError):
        StreamlitAppConfig.load_app_config(config_path=config_path)


def test_invalid_vector_store_type(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
APP_CONFIG = {"response": lambda: None, "system_vector_store": [1, 2]}
""",
    )

    with pytest.raises(ValidationError):
        StreamlitAppConfig.load_app_config(config_path=config_path)


def test_vector_store_normalization_returns_copy(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
from openai_sdk_helpers.response.base import ResponseBase
from openai_sdk_helpers.structure.base import BaseStructure


class TempResponse(ResponseBase[BaseStructure]):
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


APP_CONFIG = {"response": TempResponse, "system_vector_store": "files"}
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
from openai_sdk_helpers.response.base import ResponseBase
from openai_sdk_helpers.structure.base import BaseStructure


class TempResponse(ResponseBase[BaseStructure]):
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


APP_CONFIG = {"response": TempResponse}
""",
    )

    config = load_app_config(config_path=config_path)

    assert isinstance(config, StreamlitAppConfig)


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

    response_instance = config.create_response()

    assert isinstance(response_instance, _DummyResponse)


def test_config_accepts_response_alias(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
from openai_sdk_helpers.response.base import ResponseBase
from openai_sdk_helpers.structure.base import BaseStructure


class TempResponse(ResponseBase[BaseStructure]):
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


APP_CONFIG = {"response": TempResponse, "display_title": "Alias title"}
""",
    )

    config = StreamlitAppConfig.load_app_config(config_path=config_path)

    assert config.display_title == "Alias title"
    assert isinstance(config.create_response(), ResponseBase)


def test_config_accepts_response_class_directly(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
from openai_sdk_helpers.response.base import ResponseBase
from openai_sdk_helpers.structure.base import BaseStructure


class TempResponse(ResponseBase[BaseStructure]):
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


APP_CONFIG = TempResponse
""",
    )

    config = StreamlitAppConfig.load_app_config(config_path=config_path)

    response_instance = config.create_response()

    assert isinstance(response_instance, ResponseBase)
    assert response_instance.__class__.__name__ == "TempResponse"


def test_config_accepts_response_instance(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
from openai_sdk_helpers.response.base import ResponseBase
from openai_sdk_helpers.structure.base import BaseStructure


class TempResponse(ResponseBase[BaseStructure]):
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


APP_CONFIG = TempResponse()
""",
    )

    config = StreamlitAppConfig.load_app_config(config_path=config_path)

    response_instance = config.create_response()

    assert isinstance(response_instance, ResponseBase)
    assert response_instance.__class__.__name__ == "TempResponse"


def test_response_callable_return_type_error(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
APP_CONFIG = {"response": lambda: "bad"}
""",
    )

    config = StreamlitAppConfig.load_app_config(config_path=config_path)

    with pytest.raises(TypeError):
        config.create_response()
