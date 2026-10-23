"""Validate local settings without initializing desktop APIs."""

import json
from typing import Any, Dict, Tuple

from .config import ROOT
from .desktop import virtual_key

SETTINGS_PATH = ROOT / "settings.json"


def load_settings() -> Tuple[Dict[str, Any], Dict[str, int]]:
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or not isinstance(settings.get("keys"), dict):
        raise ValueError("settings.json must be an object containing a keys object.")
    if any(not isinstance(key, str) for key in settings["keys"].values()):
        raise ValueError("Every key binding in settings.json must be a string.")
    codes = {name: virtual_key(key) for name, key in settings["keys"].items()}
    required_keys = {"accelerate", "brake", "handbrake", "left", "right", "toggle", "stop"}
    if set(codes) != required_keys:
        raise ValueError(
            "settings.json must define accelerate, brake, handbrake, left, right, toggle, and stop keys."
        )
    if len(set(codes.values())) != len(codes):
        raise ValueError("Every key in settings.json must be different.")
    if not isinstance(settings["window_title"], str) or not settings["window_title"].strip():
        raise ValueError("window_title must be a nonempty string.")
    for name, minimum, maximum in (
        ("color_tolerance", 1, 255),
        ("car_x_fraction", 0, 1),
        ("throttle_duty", 0, 1),
    ):
        value = settings[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"{name} must be a number between {minimum} and {maximum}.")
    sample = settings["road_sample_lab"]
    if sample is not None and (
        not isinstance(sample, list)
        or len(sample) != 3
        or any(
            isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255
            for channel in sample
        )
    ):
        raise ValueError(
            "road_sample_lab must be null or three integer LAB channels between 0 and 255."
        )
    return settings, codes
