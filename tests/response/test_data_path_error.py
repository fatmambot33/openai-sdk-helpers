import pytest
from openai_sdk_helpers.response.base import BaseResponse


def test_data_path_error(openai_settings):
    """Test that data_path property raises RuntimeError if not configured."""
    r = BaseResponse(
        instructions="hi",
        tools=[],
        schema=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=openai_settings,
    )
    with pytest.raises(RuntimeError, match="data_path_fn and module_name are required"):
        _ = r.data_path
