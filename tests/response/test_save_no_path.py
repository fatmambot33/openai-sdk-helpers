from openai_sdk_helpers.response.base import BaseResponse


def test_save_skips_without_path(caplog):
    """Test that save() does not fail and logs when no path is configured."""
    r = BaseResponse(
        instructions="hi",
        tools=[],
        schema=None,
        output_structure=None,
        tool_handlers={},
        model="gpt-3",
        api_key="dummy",
    )
    caplog.set_level("DEBUG")
    r.save()  # Should log and return without error
    assert any("Skipping save" in m for m in caplog.messages)
