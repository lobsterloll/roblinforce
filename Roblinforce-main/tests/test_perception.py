"""Synthetic image fixtures exercise game-specific vision without capturing a screen."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from roblinforce import perception


def color(hue, saturation, brightness):
    return cv2.cvtColor(np.uint8([[[hue, saturation, brightness]]]), cv2.COLOR_HSV2BGR)[0, 0]


class PerceptionTests(unittest.TestCase):
    def test_road_detector_tracks_neutral_strip_and_rejects_grass(self):
        frame = np.full((800, 1000, 3), color(60, 240, 160), dtype=np.uint8)
        frame[:, 200:800] = 130
        sample = cv2.cvtColor(np.uint8([[[130, 130, 130]]]), cv2.COLOR_BGR2LAB)[0, 0].tolist()
        settings = {"road_sample_lab": sample, "car_x_fraction": 0.5}
        road = perception.screen_road(frame, settings)
        self.assertIsNotNone(road)
        self.assertAlmostEqual(road[0], 500, delta=3)
        self.assertEqual(len(road[1]), 4)
        frame[:] = color(60, 240, 160)
        self.assertIsNone(perception.screen_road(frame, settings))

    def test_road_detector_bridges_vehicle_occlusion(self):
        frame = np.full((800, 1000, 3), color(60, 240, 160), dtype=np.uint8)
        frame[:, 200:800] = 130
        frame[:, 460:540] = color(0, 240, 160)
        sample = cv2.cvtColor(np.uint8([[[130, 130, 130]]]), cv2.COLOR_BGR2LAB)[0, 0].tolist()
        road = perception.screen_road(frame, {"road_sample_lab": sample, "car_x_fraction": 0.5})
        self.assertIsNotNone(road)
        self.assertAlmostEqual(road[0], 500, delta=3)

    def test_speed_hud_absent_or_ambiguous_returns_unknown(self):
        frame = np.zeros((800, 1000, 3), dtype=np.uint8)
        self.assertFalse(perception.speed_hud_visible(frame))
        self.assertIsNone(perception.speed_moving(frame))
        frame[760:780, 740:800] = color(95, 200, 200)
        self.assertTrue(perception.speed_hud_visible(frame))
        self.assertIsNone(perception.speed_moving(frame))

    def test_two_speed_digits_mean_motion_and_single_zero_means_stopped(self):
        frame = np.zeros((1078, 1600, 3), dtype=np.uint8)
        frame[1020:1040, 1200:1300] = color(95, 200, 200)
        frame[900:950, 1300:1320] = 255
        frame[900:950, 1340:1360] = 255
        self.assertTrue(perception.speed_moving(frame))
        frame[900:950, 1340:1360] = 0
        with patch.object(perception, "SPEED_ZERO", np.full((32, 24), 255, dtype=np.uint8)):
            self.assertFalse(perception.speed_moving(frame))
        with patch.object(perception, "SPEED_ZERO", np.zeros((32, 24), dtype=np.uint8)):
            self.assertTrue(perception.speed_moving(frame))

    def test_vehicle_menu_card_positions_and_category_icon(self):
        frame = np.zeros((800, 1000, 3), dtype=np.uint8)
        self.assertIsNone(perception.car_tile(frame))
        self.assertIsNone(perception.car_category_icon(frame))
        frame[40:72, 680:940] = color(135, 180, 150)
        first = perception.car_tile(frame)
        second = perception.car_tile(frame, 1)
        self.assertIsNotNone(first)
        self.assertGreater(second[0], first[0])
        self.assertEqual(perception.car_tile(frame, 4), first)
        frame[:] = 0
        frame[80:145, 900:980] = color(110, 200, 200)
        icon = perception.car_category_icon(frame)
        self.assertIsNotNone(icon)
        self.assertGreater(icon[1], 145)


if __name__ == "__main__":
    unittest.main()
