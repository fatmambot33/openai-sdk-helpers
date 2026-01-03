def test_dummy_for_coverage(openai_settings):
    """Dummy test to increase coverage by exercising __repr__."""
    from openai_sdk_helpers.response.base import BaseResponse

    class DummyStruct:
        pass

    r = BaseResponse(
        instructions="hi",
        tools=[],
        output_structure=None,
        tool_handlers={},
        openai_settings=openai_settings,
    )
    assert "BaseResponse" in repr(r)
