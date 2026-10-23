"""Recorded-data and interruptible recovery checks with simulated resources."""

from contextlib import ExitStack
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from roblinforce import imitation, learning, recovery


class RecordingTests(unittest.TestCase):
    def test_empty_and_nonempty_recordings_preserve_npz_schema(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(imitation, "DATA", Path(directory) / "new-data"),
        ):
            self.assertIsNone(imitation.save_samples([], []))
            path = imitation.save_samples([[1, 2], [3, 4]], [-1, 1])
            with np.load(path, allow_pickle=False) as samples:
                self.assertEqual(samples["x"].dtype, np.float32)
                self.assertEqual(samples["y"].dtype, np.int8)
                self.assertEqual(samples["y"].tolist(), [-1, 1])

    def test_training_balances_classes_and_caps_reservoir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples = np.tile(np.array([[1, 2, 3]], dtype=np.float32), (4506, 1))
            labels = np.repeat([-1, 0, 1], 1502)
            np.savez(root / "drive-session.npz", x=examples, y=labels)
            with (
                patch.object(imitation, "DATA", root),
                patch.object(imitation, "MODEL", root / "models" / "model.npz"),
            ):
                imitation.train()
            with np.load(root / "models" / "model.npz", allow_pickle=False) as model:
                self.assertEqual(model["x"].shape, (4500, 3))
                np.testing.assert_array_equal(
                    np.unique(model["y"], return_counts=True)[1], [1500, 1500, 1500]
                )

    def test_training_requires_data_and_minimum_class_examples(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(imitation, "DATA", Path(directory)),
        ):
            with self.assertRaisesRegex(RuntimeError, "No recordings"):
                imitation.train()
            np.savez(Path(directory) / "drive-session.npz", x=np.ones((3, 2)), y=[-1, 0, 1])
            with self.assertRaisesRegex(RuntimeError, "at least 20"):
                imitation.train()

    def test_recording_batches_samples_and_closes_listener(self):
        keys = SimpleNamespace(stop=threading.Event(), start=Mock(), close=Mock())
        count = 0

        def health():
            nonlocal count
            count += 1
            if count >= 1001:
                keys.stop.set()

        keys.check_health = health
        capture = Mock(__enter__=Mock(return_value=object()), __exit__=Mock(return_value=False))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            patches = [
                patch.object(imitation, "DATA", Path(directory)),
                patch.object(imitation.desktop, "initialize_desktop"),
                patch.object(
                    imitation,
                    "load_settings",
                    return_value=(
                        {"window_title": "Roblox"},
                        {"left": 65, "right": 68, "accelerate": 87},
                    ),
                ),
                patch.object(imitation.desktop, "find_game_window", return_value=7),
                patch.object(imitation.desktop, "PhysicalKeys", return_value=keys),
                patch.object(imitation.desktop, "client_capture_area", return_value={"width": 100}),
                patch.object(imitation.desktop, "is_game_focused", return_value=True),
                patch.object(imitation.desktop, "physical_down", side_effect=lambda code: code == 87),
                patch.object(
                    imitation.desktop,
                    "capture_frame",
                    return_value=np.zeros((10, 10, 3), dtype=np.uint8),
                ),
                patch.object(imitation, "features", return_value=np.zeros(4, dtype=np.float32)),
                patch.object(imitation.mss, "MSS", return_value=capture),
                patch.object(imitation.time, "sleep"),
            ]
            for item in patches:
                stack.enter_context(item)
            imitation.record()
            self.assertEqual(len(list(Path(directory).glob("drive-*.npz"))), 2)
            keys.close.assert_called_once()

    def test_recording_requires_window(self):
        with (
            patch.object(imitation.desktop, "initialize_desktop"),
            patch.object(
                imitation, "load_settings", return_value=({"window_title": "Roblox"}, {})
            ),
            patch.object(imitation.desktop, "find_game_window", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "before recording"):
                imitation.record()

    def test_tabular_checkpoint_uses_atomic_final_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "models" / "steering.npz"
            with patch.object(learning, "STEERING_POLICY", path):
                learning.save_policy(np.ones((9, 5, 3)), np.zeros((9, 5, 3)))
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix(".tmp").exists())


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.keys = SimpleNamespace(
            stop=threading.Event(),
            toggle=threading.Event(),
            takeover=threading.Event(),
            check_health=Mock(),
        )
        self.output = SimpleNamespace(release_all=Mock(), set_pressed=Mock())
        self.area = {"left": 0, "top": 0, "width": 1000, "height": 800}

    def test_pointer_helpers_reject_interruption_and_forward_native_coordinates(self):
        with (
            patch.object(recovery.desktop, "is_game_focused", return_value=True),
            patch.object(recovery.desktop, "click_game_point", return_value=True) as click,
            patch.object(recovery.desktop, "scroll_game_point", return_value=True) as scroll,
        ):
            self.assertTrue(recovery.click_game(7, self.area, (50, 40), self.keys, "right"))
            self.assertTrue(recovery.scroll_game(7, self.area, (50, 40), self.keys, 120))
            click.assert_called_once_with(7, self.area, (50, 40), "right")
            scroll.assert_called_once_with(7, self.area, (50, 40), 120)
            self.keys.stop.set()
            self.assertFalse(recovery.click_game(7, self.area, (50, 40), self.keys))
            self.assertFalse(recovery.scroll_game(7, self.area, (50, 40), self.keys, 120))

    def test_respawn_checks_interruption_before_each_key(self):
        with (
            patch.object(recovery.desktop, "is_game_focused", return_value=True),
            patch.object(recovery.time, "sleep"),
        ):
            self.assertTrue(recovery.respawn_character(7, self.keys, self.output))
            self.assertEqual(
                [call.args[0] for call in self.output.set_pressed.call_args_list],
                [{0x1B}, {ord("R")}, {0x0D}],
            )
            self.keys.takeover.set()
            self.assertFalse(recovery.respawn_character(7, self.keys, self.output))

    def test_recovery_opens_menu_selects_category_and_confirms_spawn(self):
        clock = SimpleNamespace(now=0, captures=0)

        def now():
            clock.now += 0.1
            return clock.now

        def capture(*_args):
            clock.captures += 1
            return np.zeros((10, 10, 3), dtype=np.uint8)

        def tile(_frame, choice=0):
            return (50 + choice, 40) if clock.captures == 5 else None

        with ExitStack() as stack:
            patches = [
                patch.object(recovery.desktop, "is_game_focused", return_value=True),
                patch.object(recovery.desktop, "client_capture_area", return_value=self.area),
                patch.object(recovery.desktop, "capture_frame", side_effect=capture),
                patch.object(recovery, "car_tile", side_effect=tile),
                patch.object(
                    recovery,
                    "car_category_icon",
                    side_effect=lambda _frame: (20, 30) if clock.captures in (3, 4) else None,
                ),
                patch.object(
                    recovery,
                    "speed_moving",
                    side_effect=lambda _frame: None if clock.captures == 2 else True,
                ),
                patch.object(recovery, "click_game", return_value=True),
                patch.object(recovery.time, "monotonic", side_effect=now),
                patch.object(recovery.time, "sleep"),
            ]
            for item in patches:
                stack.enter_context(item)
            self.assertTrue(recovery.recover_car(object(), 7, self.keys, self.output, choice=2))
            self.assertEqual(recovery.click_game.call_count, 2)
            self.assertGreaterEqual(clock.captures, 7)
            self.output.release_all.assert_called()

    def test_recovery_menu_timeout_and_window_loss_fail_safely(self):
        clock = SimpleNamespace(now=0)

        def now():
            clock.now += 0.5
            return clock.now

        with (
            patch.object(recovery.desktop, "is_game_focused", return_value=True),
            patch.object(recovery.desktop, "client_capture_area", return_value=self.area),
            patch.object(
                recovery.desktop, "capture_frame", return_value=np.zeros((10, 10, 3), dtype=np.uint8)
            ),
            patch.object(recovery, "car_tile", return_value=(1, 2)),
            patch.object(recovery, "car_category_icon", return_value=None),
            patch.object(recovery, "click_game", return_value=True),
            patch.object(recovery.time, "monotonic", side_effect=now),
            patch.object(recovery.time, "sleep"),
        ):
            self.assertFalse(recovery.recover_car(object(), 7, self.keys, self.output))
            self.output.set_pressed.assert_called()
        with patch.object(recovery.desktop, "is_game_focused", return_value=False):
            self.assertFalse(recovery.recover_car(object(), 7, self.keys, self.output))
        with (
            patch.object(recovery.desktop, "is_game_focused", return_value=True),
            patch.object(recovery.desktop, "client_capture_area", return_value=None),
        ):
            self.assertFalse(recovery.recover_car(object(), 7, self.keys, self.output))

    def test_recovery_aborts_when_listener_is_unhealthy(self):
        self.keys.check_health.side_effect = RuntimeError("listener failed")
        with self.assertRaisesRegex(RuntimeError, "listener failed"):
            recovery.recover_car(object(), 7, self.keys, self.output)
        self.output.release_all.assert_called_once()


if __name__ == "__main__":
    unittest.main()
