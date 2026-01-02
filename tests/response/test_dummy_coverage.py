def test_dummy_for_coverage():
    """Dummy test to increase coverage by exercising __repr__."""
    from openai_sdk_helpers.response.base import BaseResponse

    class DummyStruct:
        pass

    r = BaseResponse(
        instructions="hi",
        tools=[],
        schema=None,
        output_structure=None,
        tool_handlers={},
        model="gpt-3",
        api_key="dummy",
    )
    assert "BaseResponse" in repr(r)
