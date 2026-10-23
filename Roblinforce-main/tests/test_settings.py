"""Settings validation remains independent of desktop permissions and hooks."""

import json
from unittest.mock import Mock

import pytest

from roblinforce import desktop, settings as settings_module
from roblinforce.config import ROOT


@pytest.fixture
def settings_path(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    path.write_text((ROOT / "settings.json").read_text())
    monkeypatch.setattr(settings_module, "SETTINGS_PATH", path)
    return path


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("window_title", "", "nonempty"),
        ("window_title", 123, "nonempty"),
        ("color_tolerance", 0, "color_tolerance"),
        ("color_tolerance", True, "color_tolerance"),
        ("car_x_fraction", 1.1, "car_x_fraction"),
        ("throttle_duty", float("nan"), "throttle_duty"),
        ("road_sample_lab", [0, 0], "LAB"),
        ("road_sample_lab", [0, False, 0], "LAB"),
        ("road_sample_lab", [0, 256, 0], "LAB"),
    ],
)
def test_invalid_settings_are_rejected(settings_path, field, value, message):
    settings = json.loads(settings_path.read_text())
    settings_path.write_text(json.dumps({**settings, field: value}))
    with pytest.raises(ValueError, match=message):
        settings_module.load_settings()


def test_key_bindings_must_be_complete_and_distinct(settings_path):
    settings = json.loads(settings_path.read_text())
    settings["keys"]["right"] = settings["keys"]["left"]
    settings_path.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="different"):
        settings_module.load_settings()
    del settings["keys"]["right"]
    settings_path.write_text(json.dumps(settings))
    with pytest.raises(ValueError, match="must define"):
        settings_module.load_settings()


def test_settings_load_does_not_initialize_native_api(settings_path, monkeypatch):
    monkeypatch.setattr(desktop, "initialize_desktop", Mock(side_effect=AssertionError))
    settings, codes = settings_module.load_settings()
    assert settings["window_title"] == "Roblox"
    assert codes["accelerate"] == ord("W")


def test_settings_reject_non_object_and_non_string_keys(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(settings_module, "SETTINGS_PATH", path)
    for invalid in ([], {"keys": None}, {"keys": {"accelerate": 1}}):
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            settings_module.load_settings()


def test_paths_and_settings_are_independent_of_working_directory(tmp_path, monkeypatch):
    from pathlib import Path

    from roblinforce import config, perception

    monkeypatch.chdir(tmp_path)
    settings, _codes = settings_module.load_settings()
    repository = Path(__file__).resolve().parents[1]
    assert config.ROOT == repository
    assert config.DATA == repository / "data"
    assert config.MODELS == repository / "models"
    assert settings_module.SETTINGS_PATH == repository / "settings.json"
    assert config.ASSETS == repository / "src" / "roblinforce" / "assets"
    assert perception.SPEED_ZERO is not None
    assert settings["window_title"] == "Roblox"


@pytest.mark.parametrize("module_name", ["visual", "imitation"])
def test_uncalibrated_driving_fails_before_desktop_initialization(monkeypatch, module_name):
    from roblinforce import imitation, visual

    module = {"visual": visual, "imitation": imitation}[module_name]
    monkeypatch.setattr(module, "load_settings", lambda: ({"road_sample_lab": None}, {}))
    monkeypatch.setattr(desktop, "initialize_desktop", Mock(side_effect=AssertionError("desktop")))
    with pytest.raises(RuntimeError, match="roblinforce calibrate"):
        if module is visual:
            module.ai_drive()
        else:
            module.drive(self_training=True)
