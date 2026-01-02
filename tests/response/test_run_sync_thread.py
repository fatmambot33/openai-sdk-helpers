import pytest
from openai_sdk_helpers.response.base import BaseResponse

pytest.skip("Skip test: requires valid OpenAI API key.", allow_module_level=True)


def test_run_sync_thread(openai_settings):
    """Test run_sync fallback thread path for coverage."""
    r = BaseResponse(
        instructions="hi",
        tools=[],
        schema=None,
        output_structure=None,
        tool_handlers={},
        openai_settings=openai_settings,
    )
    # This will hit the thread fallback path
    result = r.run_sync("hello")
    assert result is None
