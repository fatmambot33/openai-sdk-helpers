import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

# Import normally instead of dynamic loading
from openai_sdk_helpers.environment import get_data_path
from openai_sdk_helpers.utils.coercion import ensure_list
from openai_sdk_helpers.utils.json import (
    DataclassJSONSerializable,
    coerce_jsonable,
    customJSONEncoder,
)
from openai_sdk_helpers.utils.path_utils import check_filepath
from openai_sdk_helpers.logging import log


def test_ensure_list_behavior():
    assert ensure_list(None) == []
    assert ensure_list(1) == [1]
    assert ensure_list([1, 2]) == [1, 2]
    assert ensure_list((1, 2)) == [1, 2]


def test_check_filepath_creates_parent(tmp_path):
    target = tmp_path / "sub" / "file.txt"
    res = check_filepath(filepath=target)
    assert res == target
    assert target.parent.exists()


def test_json_serializable_and_encoder(tmp_path):
    class Color(Enum):
        RED = "red"

    @dataclass
    class Dummy(DataclassJSONSerializable):
        name: str = "x"
        path: Path = Path("a/b")
        color: Color = Color.RED
        ts: datetime = datetime(2020, 1, 1)

    d = Dummy()
    j = d.to_json()
    assert j["name"] == "x"

    out = tmp_path / "out.json"
    path_str = d.to_json_file(out)
    assert Path(path_str).exists()

    payload = {"p": Path("x/y"), "c": Color.RED, "ts": datetime(2020, 1, 1)}
    s = json.dumps(payload, cls=customJSONEncoder)
    assert "red" in s


def test_get_data_path_monkeypatched(monkeypatch, tmp_path):
    # override home to avoid writing to real user home
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    p = get_data_path("mymod")
    assert p.exists()
    assert p.name == "mymod"


def test_log_runs():
    # ensure the log helper does not raise
    log("testing log")


def test_custom_json_encoder_handles_sets_and_models(tmp_path):
    class DummyModel:
        def model_dump(self) -> dict[str, object]:
            return {"numbers": {1, 2}, "path": Path("/tmp/example")}

    encoded = customJSONEncoder().encode(DummyModel())
    assert "example" in encoded
    assert "numbers" in encoded


def test_log_is_idempotent(caplog):
    caplog.set_level("INFO")
    log("first")
    log("second")
    assert any(record.message == "first" for record in caplog.records)
    assert any(record.message == "second" for record in caplog.records)


def test_coerce_jsonable_serializes_structures_and_dataclasses():
    from openai_sdk_helpers.structure.base import StructureBase

    class ExampleStructure(StructureBase):
        message: str

    @dataclass
    class Wrapper:
        content: Path

    payload = {
        "structure": ExampleStructure(message="hello"),
        "wrapper": Wrapper(content=Path("a/b")),
    }

    serialized = coerce_jsonable(payload)

    assert serialized["structure"]["message"] == "hello"
    assert serialized["wrapper"]["content"] == "a/b"
    assert json.dumps(serialized)


def test_to_jsonable_exports():
    """Test that to_jsonable is properly exported."""
    from openai_sdk_helpers.utils import to_jsonable

    result = to_jsonable({"key": "value"})
    assert result == {"key": "value"}


def test_reference_encoding_helpers():
    """Test get_module_qualname, encode_module_qualname, decode_module_qualname."""
    from openai_sdk_helpers.utils import (
        decode_module_qualname,
        encode_module_qualname,
        get_module_qualname,
    )

    # Test with Path class
    result = get_module_qualname(Path)
    assert result is not None
    # Module could be "pathlib" or "pathlib._local" depending on Python version
    assert result[0].startswith("pathlib")
    assert result[1] == "Path"

    # Test encoding
    encoded = encode_module_qualname(Path)
    assert encoded is not None
    assert encoded["module"].startswith("pathlib")
    assert encoded["qualname"] == "Path"

    # Test decoding - use the actual encoded module for round-trip test
    decoded = decode_module_qualname(encoded)
    assert decoded is Path

    # Test with invalid input
    assert decode_module_qualname({}) is None
    assert decode_module_qualname({"module": "fake_module"}) is None


def test_StructureBase_class_encoding():
    """Test that StructureBase classes are encoded with __structure_class__ marker."""
    from openai_sdk_helpers.structure.base import StructureBase
    from openai_sdk_helpers.utils import to_jsonable

    class TestStructure(StructureBase):
        value: str

    # Test encoding of the class (not instance)
    encoded = to_jsonable(TestStructure)
    assert isinstance(encoded, dict)
    assert encoded.get("__structure_class__") is True
    assert "module" in encoded
    assert "qualname" in encoded


def test_dataclass_json_serializable_alias():
    """Test that DataclassJSONSerializable is properly aliased."""
    from openai_sdk_helpers.utils import DataclassJSONSerializable

    @dataclass
    class TestData(DataclassJSONSerializable):
        name: str
        count: int

    instance = TestData(name="test", count=5)
    json_data = instance.to_json()
    assert json_data["name"] == "test"
    assert json_data["count"] == 5

    # Test round-trip
    restored = TestData.from_json(json_data)
    assert restored.name == "test"
    assert restored.count == 5


def test_basemodel_json_serializable(tmp_path, caplog):
    """Test BaseModelJSONSerializable for Pydantic models."""
    import logging

    from pydantic import BaseModel
    from openai_sdk_helpers.utils import BaseModelJSONSerializable

    class Color(Enum):
        RED = "red"

    class TestModel(BaseModelJSONSerializable, BaseModel):
        name: str
        value: int
        color: Color | None = None

    # Test to_json
    instance = TestModel(name="test", value=42, color=Color.RED)
    json_data = instance.to_json()
    assert json_data["name"] == "test"
    assert json_data["value"] == 42
    assert json_data["color"] is Color.RED

    # Test from_json
    restored = TestModel.from_json(json_data)
    assert restored.name == "test"
    assert restored.value == 42
    assert restored.color is Color.RED

    assert (
        TestModel.from_json({"name": "test", "value": 1, "color": "RED"}).color
        is Color.RED
    )
    assert (
        TestModel.from_json({"name": "test", "value": 1, "color": "red"}).color
        is Color.RED
    )

    caplog.set_level(logging.WARNING, logger="openai_sdk_helpers")
    restored_none = TestModel.from_json({"name": "test", "value": 1, "color": "blue"})
    assert restored_none.color is None
    assert any(
        "Invalid value for 'color'" in record.message for record in caplog.records
    )

    # Test file I/O
    filepath = tmp_path / "model.json"
    saved_path = instance.to_json_file(filepath)
    assert Path(saved_path).exists()

    loaded = TestModel.from_json_file(filepath)
    assert loaded.name == "test"
    assert loaded.value == 42


def test_to_jsonable_with_sets():
    """Test that sets are converted to lists."""
    from openai_sdk_helpers.utils import to_jsonable

    data = {"numbers": {1, 2, 3}}
    result = to_jsonable(data)
    assert isinstance(result["numbers"], list)
    assert set(result["numbers"]) == {1, 2, 3}


def test_encoder_with_datetime():
    """Test customJSONEncoder with datetime."""
    from openai_sdk_helpers.utils import customJSONEncoder

    dt = datetime(2023, 1, 15, 10, 30, 45)
    encoded = json.dumps({"timestamp": dt}, cls=customJSONEncoder)
    data = json.loads(encoded)
    assert "2023-01-15" in data["timestamp"]
    assert "10:30:45" in data["timestamp"]


def test_path_deserialization_in_dataclass(tmp_path):
    """Test that Path fields are correctly deserialized."""
    from openai_sdk_helpers.utils import DataclassJSONSerializable

    @dataclass
    class PathData(DataclassJSONSerializable):
        file_path: Path
        name: str

    json_data = {"file_path": "/tmp/test.txt", "name": "test"}
    instance = PathData.from_json(json_data)
    assert isinstance(instance.file_path, Path)
    assert str(instance.file_path) == "/tmp/test.txt"
    assert instance.name == "test"


def test_private_properties_not_serialized():
    """Test that private properties (starting with underscore) are not serialized."""
    from openai_sdk_helpers.utils import to_jsonable

    # Test with dict
    data = {"public": "visible", "_private": "hidden", "__very_private": "also_hidden"}
    result = to_jsonable(data)
    assert "public" in result
    assert "_private" not in result
    assert "__very_private" not in result

    # Test with object __dict__
    class TestObj:
        def __init__(self):
            self.public = "visible"
            self._private = "hidden"

    obj = TestObj()
    result = to_jsonable(obj.__dict__)
    assert "public" in result
    assert "_private" not in result
