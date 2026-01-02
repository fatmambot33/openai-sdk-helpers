from openai_sdk_helpers.response.base import BaseResponse
import pytest

import pytest

pytest.skip("Skip test: requires valid OpenAI API key.", allow_module_level=True)


def test_run_sync_thread():
    """Test run_sync fallback thread path for coverage."""
    r = BaseResponse(
        instructions="hi",
        tools=[],
        schema=None,
        output_structure=None,
        tool_handlers={},
        model="gpt-3",
        api_key="dummy",
    )
    # This will hit the thread fallback path
    result = r.run_sync("hello")
    assert result is None
