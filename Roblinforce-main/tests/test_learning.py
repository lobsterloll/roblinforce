"""Regression checks for offline learning; these never send desktop input."""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from roblinforce.learning import (
    VisualLearner,
    predict,
    road_state,
    train_visual,
    update_policy,
    visual_values,
)
from roblinforce.perception import features, scene_moving, visual_state


class LearningTests(unittest.TestCase):
    def test_visual_state_keeps_checkpoint_input_dimensions(self):
        frame = np.full((120, 200, 3), 255, dtype=np.uint8)
        road = visual_state(frame, False)
        explorer = visual_state(frame, True, full_view=True)
        self.assertEqual(road.shape, (3601,))
        self.assertEqual(explorer.shape, (9217,))
        np.testing.assert_allclose(road[:-1], 0.5)
        self.assertEqual(explorer[-1], 1)

    def test_road_state_centered_road_has_neutral_steering(self):
        road = (100.0, [(80, 100.0), (40, 100.0)])
        self.assertEqual(road_state(road, 200, 0.5), ((4, 2), 0, 0, 0))

    def test_nearest_neighbor_prediction_preserves_vote_and_similarity(self):
        examples = np.eye(8, dtype=np.float32)
        direction, confidence, agreement = predict(examples[0], examples, np.ones(8, dtype=np.int8))
        self.assertEqual((direction, confidence, agreement), (1, 1.0, 7))

    def test_tabular_update_returns_new_arrays(self):
        values = np.zeros((9, 5, 3), dtype=np.float32)
        visits = np.zeros_like(values, dtype=np.int32)
        updated, counts = update_policy(values, visits, (4, 2), 0, 1.0)
        self.assertAlmostEqual(float(updated[4, 2, 1]), 0.2)
        self.assertEqual(counts[4, 2, 1], 1)
        self.assertEqual(values.sum(), 0)
        self.assertEqual(visits.sum(), 0)

    def test_visual_training_does_not_mutate_weights(self):
        weights = tuple(np.zeros(shape, dtype=np.float32) for shape in ((2, 3), (3,), (3, 2), (2,)))
        state = np.ones(2, dtype=np.float32)
        replay = [(state, 1, 1.0, state, True)] * 32
        updated = train_visual(weights, weights, replay, random.Random(7))
        self.assertGreater(updated[-1][1], 0)
        self.assertEqual(weights[-1][1], 0)
        np.testing.assert_array_equal(visual_values(state, weights), [0, 0])

    def test_checkpoint_round_trip_and_shape_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models" / "agent.npz"
            learner = VisualLearner.load(path, 2, 3)
            learner.save(path)
            restored = VisualLearner.load(path, 2, 3)
            self.assertEqual(restored.steps, 0)
            for actual, expected in zip(restored.weights, learner.weights):
                np.testing.assert_array_equal(actual, expected)
            with self.assertRaisesRegex(RuntimeError, "invalid shape"):
                VisualLearner.load(path, 4, 3)

    def test_feature_and_motion_boundaries(self):
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        self.assertTrue(np.isfinite(features(frame)).all())
        self.assertFalse(scene_moving(None, frame))
        self.assertFalse(scene_moving(frame, frame))
        self.assertTrue(scene_moving(frame, np.full_like(frame, 255)))


if __name__ == "__main__":
    unittest.main()
