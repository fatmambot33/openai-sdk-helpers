import pytest

from openai_sdk_helpers.enums.base import CrosswalkJSONEnum


def test_crosswalkjsonenum_requires_override():
    class SampleEnum(CrosswalkJSONEnum):
        EXAMPLE = "example"

    with pytest.raises(NotImplementedError):
        SampleEnum.CROSSWALK()
