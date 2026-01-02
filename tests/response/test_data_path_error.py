import pytest
from openai_sdk_helpers.response.base import BaseResponse


def test_data_path_error():
    """Test that data_path property raises RuntimeError if not configured."""
    r = BaseResponse(
        instructions="hi",
        tools=[],
        schema=None,
        output_structure=None,
        tool_handlers={},
        model="gpt-3",
        api_key="dummy",
    )
    with pytest.raises(RuntimeError, match="data_path_fn and module_name are required"):
        _ = r.data_path
