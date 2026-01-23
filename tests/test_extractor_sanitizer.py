from openai_sdk_helpers.extract.extractor import _SanitizingFormatHandler


def test_sanitizing_format_handler_drops_nulls_and_coerces_values() -> None:
    handler = _SanitizingFormatHandler()
    text = """```json
    {
      "extractions": [
        {
          "person": "Ada Lovelace",
          "company": null,
          "notes": ["first", "programmer"],
          "extra": {"source": "bio"},
          "person_attributes": {"source": "resume"},
          "company_attributes": ["invalid"]
        }
      ]
    }
    ```"""

    items = handler.parse_output(text)

    assert items == [
        {
            "person": "Ada Lovelace",
            "notes": '["first", "programmer"]',
            "extra": '{"source": "bio"}',
            "person_attributes": {"source": "resume"},
        }
    ]
