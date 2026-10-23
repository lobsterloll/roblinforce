"""Calibration samples preview pixels without ever injecting desktop input."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import cv2
import numpy as np
import pytest

from roblinforce import calibration, desktop


def test_calibration_returns_new_settings_without_mutating_existing():
    settings = {"road_sample_lab": None}
    calibrated, status = calibration.calibrate_road(np.full((17, 17, 3), 90, np.uint8), settings)
    assert settings == {"road_sample_lab": None}
    assert len(calibrated["road_sample_lab"]) == 3
    assert status.startswith("CALIBRATED")


@pytest.fixture
def calibration_preview(tmp_path, monkeypatch):
    settings = {"window_title": "Roblox", "road_sample_lab": None, "car_x_fraction": 0.5}
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(settings))
    state = SimpleNamespace(
        frame=np.full((100, 100, 3), 90, np.uint8),
        area={"left": 0, "top": 0, "width": 100, "height": 100},
        point=(25, 25),
        ticks=0,
        callback=None,
    )

    def wait_key(_delay):
        state.ticks += 1
        if state.ticks == 1:
            state.callback(cv2.EVENT_LBUTTONDOWN, *state.point, 0, None)
            return -1
        return ord("q")

    monkeypatch.setattr(calibration, "SETTINGS_PATH", path)
    monkeypatch.setattr(calibration, "load_settings", lambda: (settings, {}))
    monkeypatch.setattr(desktop, "initialize_desktop", Mock())
    monkeypatch.setattr(desktop, "find_game_window", lambda _title: 7)
    monkeypatch.setattr(desktop, "client_capture_area", lambda _window: state.area)
    monkeypatch.setattr(desktop, "is_game_focused", lambda _window: False)
    monkeypatch.setattr(
        desktop, "capture_frame", Mock(side_effect=lambda *_args: state.frame.copy())
    )
    for operation in ("KeyOutput", "PhysicalKeys", "click_game_point", "scroll_game_point"):
        monkeypatch.setattr(desktop, operation, Mock(side_effect=AssertionError("injected input")))
    monkeypatch.setattr(calibration.mss, "MSS", MagicMock())
    monkeypatch.setattr(calibration, "screen_road", lambda *_args: None)
    monkeypatch.setattr(calibration, "draw_preview", lambda *_args: (state.frame, 0.5))
    monkeypatch.setattr(cv2, "namedWindow", Mock())
    monkeypatch.setattr(cv2, "imshow", Mock())
    monkeypatch.setattr(cv2, "destroyAllWindows", Mock())
    monkeypatch.setattr(cv2, "setMouseCallback", lambda _title, fn: setattr(state, "callback", fn))
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda *_args: 1)
    return state, path


@pytest.mark.parametrize("color, accepted", [((90, 90, 90), True), ((0, 180, 0), False)])
def test_preview_calibration_saves_only_gray_road(calibration_preview, color, accepted):
    state, path = calibration_preview
    state.frame[:] = color
    calibration.run()
    assert (json.loads(path.read_text())["road_sample_lab"] is not None) is accepted
    # Preview focus must retain the last game capture, including its coordinate scale.
    desktop.capture_frame.assert_called_once()
    cv2.destroyAllWindows.assert_called_once()


def test_preview_edge_click_does_not_save_invalid_patch(calibration_preview):
    state, path = calibration_preview
    state.point = (0, 0)
    calibration.run()
    assert json.loads(path.read_text())["road_sample_lab"] is None


def test_missing_game_reports_actionable_error(calibration_preview, monkeypatch):
    monkeypatch.setattr(desktop, "find_game_window", lambda _title: None)
    with pytest.raises(RuntimeError, match="Open Roblox"):
        calibration.run()


def test_minimized_game_closes_preview_on_failure(calibration_preview):
    state, _path = calibration_preview
    state.area = None
    with pytest.raises(RuntimeError, match="minimized"):
        calibration.run()
    cv2.destroyAllWindows.assert_called_once()


def test_preview_click_uses_capture_coordinates_after_scaling(calibration_preview):
    state, path = calibration_preview
    state.frame[42:59, 42:59] = 140
    calibration.run()
    expected = cv2.cvtColor(np.uint8([[[140, 140, 140]]]), cv2.COLOR_BGR2LAB)[0, 0]
    assert json.loads(path.read_text())["road_sample_lab"] == expected.tolist()
