"""Road geometry and preview tests use generated frames, never desktop capture."""

import cv2
import numpy as np
import pytest

from roblinforce import road


@pytest.fixture
def settings():
    sample = cv2.cvtColor(np.full((1, 1, 3), 80, np.uint8), cv2.COLOR_BGR2LAB)[0, 0]
    return {"road_sample_lab": sample.tolist(), "color_tolerance": 15, "car_x_fraction": 0.5}


def test_color_road_tracks_continuous_asphalt(settings):
    frame = np.zeros((600, 1000, 3), np.uint8)
    frame[:, 300:700] = 80
    target, centers = road.road_center(frame, settings)
    assert target == pytest.approx(500)
    assert len(centers) == 4


def test_unclear_or_overly_wide_road_is_rejected(settings):
    assert road.road_center(np.zeros((600, 1000, 3), np.uint8), settings) is None
    assert road.road_center(np.full((600, 1000, 3), 80, np.uint8), settings) is None


def test_preview_preserves_input_and_scales(settings):
    frame = np.full((600, 1200, 3), 80, np.uint8)
    before = frame.copy()
    preview, scale = road.draw_preview(frame, settings, (600, [(264, 600), (216, 600)]), "PAUSED")
    assert np.array_equal(frame, before)
    assert preview.shape == (500, 1000, 3)
    assert scale == pytest.approx(1000 / 1200)


def test_nearby_color_runs_merge():
    band = np.array([0, 1, 1, 0, 1, 1, 0, 0], np.int8)
    assert road._road_runs(band, 100) == [(1, 6)]


def test_marked_lane_needs_asphalt_support_and_correct_yellow_side(monkeypatch):
    mask = np.full((180, 1000), 255, np.uint8)
    hsv = np.zeros((180, 1000, 3), np.uint8)
    yellow = [np.array([350.0, 420.0])]
    white = [np.array([650.0, 580.0])]
    monkeypatch.setattr(
        road, "_marking_positions", lambda *args: yellow if args[-1] == 3 else white
    )
    target, centers = road._marked_lane(mask, hsv, (600, 1000), 102, 500)
    assert target == 500
    assert len(centers) == 2
    assert road._marked_lane(np.zeros_like(mask), hsv, (600, 1000), 102, 500) is None
    yellow[:] = [np.array([650.0, 580.0])]
    white[:] = [np.array([350.0, 420.0])]
    assert road._marked_lane(mask, hsv, (600, 1000), 102, 500) is None


def test_marking_clusters_require_support(monkeypatch):
    lines = np.array(
        [
            [[350, 140, 420, 80]],
            [[352, 140, 422, 80]],
            [[354, 140, 424, 80]],
            [[10, 10, 20, 10]],
            [[-900, 140, -800, 80]],
        ]
    )
    monkeypatch.setattr(cv2, "HoughLinesP", lambda *args, **kwargs: lines)
    positions = road._marking_positions(
        np.zeros((180, 1000, 3), np.uint8), (600, 1000), 102, (0, 0, 0), (179, 255, 255), 3
    )
    assert len(positions) == 1
    assert positions[0][0] == pytest.approx(347.333333)


def test_color_road_rejects_sudden_widening():
    mask = np.zeros((180, 1000), np.uint8)
    mask[140:170, 450:550] = 255
    mask[:140, 200:800] = 255
    assert road._color_road(mask, (600, 1000), 102, 500) is None
