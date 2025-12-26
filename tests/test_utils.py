import importlib.util
import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import pytest

# Load modules directly from src to avoid executing corai/__init__.py
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_CORAI = os.path.join(ROOT, "src", "corai")


def _load_module(name: str, filename: str):
    path = os.path.join(SRC_CORAI, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # register module in sys.modules so decorators (dataclass) can resolve module
    import sys

    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


env_mod = _load_module("corai.environment", "environment.py")
utils_mod = _load_module("corai.utils.core", os.path.join("utils", "core.py"))

get_data_path = env_mod.get_data_path
JSONSerializable = utils_mod.JSONSerializable
check_filepath = utils_mod.check_filepath
customJSONEncoder = utils_mod.customJSONEncoder
ensure_list = utils_mod.ensure_list
log = utils_mod.log


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
    class Dummy(JSONSerializable):
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
    monkeypatch.setattr(env_mod.Path, "home", lambda: tmp_path)
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
