"""Simulated desktop integration checks: no real windows, capture, or input."""

from contextlib import ExitStack, contextmanager
from pathlib import Path
import random
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from roblinforce import imitation, visual
from roblinforce.config import AI_ACTIONS, FREE_ACTIONS
from roblinforce.learning import VisualLearner


class FakeOutput:
    def __init__(self, window):
        self.window = window
        self.pressed = set()
        self.commands = []
        self.releases = 0

    def set_pressed(self, wanted):
        self.pressed = set(wanted)
        self.commands.append((self.window, set(wanted)))

    def release_all(self):
        self.pressed = set()
        self.releases += 1


@contextmanager
def simulated_driver(
    module,
    directory,
    *,
    frames=5,
    focus=None,
    motion=True,
    windows=None,
    health_error=False,
    capture_error=False,
    free=False,
    stop_at=None,
    takeover_at=None,
):
    clock = SimpleNamespace(frame=0, now=1.0)
    output = FakeOutput(None)
    keys = SimpleNamespace(
        stop=threading.Event(),
        takeover=threading.Event(),
        toggle=threading.Event(),
        start=Mock(),
        close=Mock(),
    )

    def check_health():
        clock.frame += 1
        clock.now += 0.7
        if clock.frame == 1 or (windows is not None and clock.frame == 2):
            keys.toggle.set()
        if stop_at == clock.frame:
            keys.stop.set()
        if takeover_at == clock.frame:
            keys.takeover.set()
        if health_error and clock.frame == 3:
            raise RuntimeError("listener failed")

    keys.check_health = check_health
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    area = {"left": 0, "top": 0, "width": 160, "height": 100}
    codes = {
        name: index + 1
        for index, name in enumerate(("accelerate", "left", "right", "brake", "handbrake"))
    }
    root = Path(directory)
    policy = root / "agent.npz"
    learner = VisualLearner.load(policy, 2, len(FREE_ACTIONS if free else AI_ACTIONS))
    learner.save(policy)

    def capture(_capture, _area):
        if capture_error and clock.frame == 2:
            raise OSError("capture failed")
        return frame.copy()

    def make_output(window):
        output.window = window
        return output

    capture_context = Mock()
    capture_context.__enter__ = Mock(return_value=object())
    capture_context.__exit__ = Mock(return_value=False)
    with ExitStack() as stack:
        patches = [
            patch.object(module.desktop, "initialize_desktop"),
            patch.object(
                module,
                "load_settings",
                return_value=(
                    {
                        "window_title": "Roblox",
                        "car_x_fraction": 0.5,
                        "throttle_duty": 0.5,
                        "road_sample_lab": [90, 128, 128],
                    },
                    codes,
                ),
            ),
            patch.object(
                module.desktop,
                "find_game_window",
                side_effect=windows if windows else lambda _title: 7,
            ),
            patch.object(
                module.desktop,
                "client_capture_area",
                side_effect=lambda window: area if window else None,
            ),
            patch.object(
                module.desktop,
                "is_game_focused",
                side_effect=lambda window: bool(window) and (focus(clock.frame) if focus else True),
            ),
            patch.object(module.desktop, "capture_frame", side_effect=capture),
            patch.object(module.desktop, "PhysicalKeys", return_value=keys),
            patch.object(module.desktop, "KeyOutput", side_effect=make_output),
            patch.object(module.mss, "MSS", return_value=capture_context),
            patch.object(module.time, "monotonic", side_effect=lambda: clock.now),
            patch.object(module.time, "sleep"),
            patch.object(module.cv2, "namedWindow"),
            patch.object(module.cv2, "imshow"),
            patch.object(module.cv2, "destroyAllWindows"),
            patch.object(
                module.cv2,
                "waitKey",
                side_effect=lambda _delay: ord("q") if clock.frame >= frames else -1,
            ),
            patch.object(
                module.cv2,
                "getWindowProperty",
                side_effect=lambda *_args: 0 if clock.frame >= frames else 1,
            ),
        ]
        if module is visual:
            patches += [
                patch.object(module, "VISUAL_POLICY", policy),
                patch.object(module, "FREE_POLICY", policy),
                patch.object(module.VisualLearner, "load", return_value=learner),
                patch.object(module, "screen_road", return_value=(80, [(60, 80), (40, 80)])),
                patch.object(module, "car_tile", return_value=None),
                patch.object(module, "car_category_icon", return_value=None),
                patch.object(module, "speed_hud_visible", return_value=True),
                patch.object(
                    module,
                    "speed_moving",
                    side_effect=motion if callable(motion) else lambda _frame: motion,
                ),
                patch.object(module, "scene_moving", return_value=True),
                patch.object(module, "visual_state", return_value=np.zeros(2, dtype=np.float32)),
                patch("roblinforce.evaluation.DATA", root),
            ]
        else:
            patches += [
                patch.object(module, "STEERING_POLICY", root / "steering.npz"),
                patch.object(module, "road_center", return_value=(80, [(60, 80), (40, 80)])),
                patch.object(module, "draw_preview", return_value=(frame, 1)),
                patch.object(module, "speed_moving", return_value=motion),
                patch.object(module, "save_policy"),
            ]
        for item in patches:
            stack.enter_context(item)
        yield SimpleNamespace(
            output=output, keys=keys, clock=clock, policy=policy, learner=learner, root=root
        )


class VisualRuntimeTests(unittest.TestCase):
    def test_training_runs_actions_and_releases_controls_on_exit(self):
        with tempfile.TemporaryDirectory() as directory, simulated_driver(visual, directory) as run:
            visual.ai_drive()
            self.assertGreater(len(run.output.commands), 0)
            self.assertEqual(run.output.pressed, set())
            run.keys.close.assert_called_once()
            with np.load(run.policy) as saved:
                self.assertGreater(int(saved["steps"]), 0)

    def test_evaluation_writes_evidence_without_changing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, simulated_driver(visual, directory) as run:
            before = run.policy.read_bytes()
            with (
                patch.object(
                    VisualLearner, "observe", side_effect=AssertionError("evaluation trained")
                ),
                patch.object(VisualLearner, "save", side_effect=AssertionError("evaluation saved")),
            ):
                visual.ai_drive(evaluation=True)
            self.assertEqual(run.policy.read_bytes(), before)
            self.assertEqual(len(list((run.root / "evaluations").glob("*/summary.json"))), 1)

    def test_free_evaluation_uses_deterministic_action_selection(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            simulated_driver(visual, directory, free=True) as run,
        ):
            visual.ai_drive(free_exploration=True, evaluation=True)
            self.assertGreater(len(run.output.commands), 0)

    def test_window_reacquisition_rebinds_output_before_start(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            simulated_driver(visual, directory, windows=[None, 42]) as run,
        ):
            visual.ai_drive()
            self.assertTrue(run.output.commands)
            self.assertTrue(all(window == 42 for window, _keys in run.output.commands))

    def test_focus_loss_capture_failure_and_physical_stop_release_output(self):
        scenarios = (
            {"focus": lambda frame: frame != 2},
            {"capture_error": True},
            {"stop_at": 3},
            {"takeover_at": 3},
        )
        for options in scenarios:
            with (
                self.subTest(options=options),
                tempfile.TemporaryDirectory() as directory,
                simulated_driver(visual, directory, **options) as run,
            ):
                visual.ai_drive()
                self.assertEqual(run.output.pressed, set())
                self.assertGreater(run.output.releases, 1)
                run.keys.close.assert_called_once()

    def test_dead_listener_aborts_and_still_cleans_up(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            simulated_driver(visual, directory, health_error=True) as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "listener failed"):
                visual.ai_drive()
            run.keys.close.assert_called_once()
            self.assertFalse(run.output.pressed)

    def test_missing_checkpoint_cannot_be_evaluated(self):
        with tempfile.TemporaryDirectory() as directory, simulated_driver(visual, directory) as run:
            run.policy.unlink()
            with self.assertRaisesRegex(RuntimeError, "Train this mode"):
                visual.ai_drive(evaluation=True)
            run.keys.start.assert_not_called()

    def test_unknown_motion_attempts_recovery_and_missing_penalty(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            simulated_driver(
                visual, directory, motion=lambda _frame: True if run.clock.frame == 1 else None
            ) as run,
        ):
            # Prevent scene motion fallback from masking an absent speed/road signal.
            with (
                patch.object(visual, "screen_road", return_value=None),
                patch.object(visual, "recover_car", return_value=False) as recover,
            ):
                visual.ai_drive()
            self.assertGreater(recover.call_count, 0)
            with np.load(run.policy) as saved:
                self.assertGreater(int(saved["steps"]), 0)


class ActionTests(unittest.TestCase):
    def test_evaluation_is_greedy_and_stopped_explorer_starts_car(self):
        learner = VisualLearner.load(Path("/does-not-exist/checkpoint.npz"), 2, len(FREE_ACTIONS))
        state = np.zeros(2, dtype=np.float32)
        self.assertEqual(
            visual.select_action(
                state, learner, FREE_ACTIONS, random.Random(7), True, True, False, True, False
            ),
            (0, "start_car"),
        )
        first = visual.select_action(
            state, learner, FREE_ACTIONS, random.Random(1), True, True, True, False, None
        )
        second = visual.select_action(
            state, learner, FREE_ACTIONS, random.Random(9), True, True, True, False, None
        )
        self.assertEqual(first, second)
        with patch.object(visual, "visual_values", return_value=np.arange(len(FREE_ACTIONS))):
            self.assertEqual(
                visual.select_action(
                    state, learner, FREE_ACTIONS, random.Random(1), True, True, True, True, True
                ),
                (2, "greedy"),
            )
        action, selection = visual.select_action(
            state,
            learner,
            FREE_ACTIONS,
            Mock(random=lambda: 0, randrange=lambda count: count - 1),
            True,
            False,
            False,
            False,
            None,
        )
        self.assertEqual((action, selection), (len(FREE_ACTIONS) - 1, "explore"))

    def test_action_dispatch_preserves_keyboard_click_and_wheel_behavior(self):
        output = FakeOutput(7)
        codes = {"accelerate": 87, "left": 65}
        area = {"width": 100, "height": 50}
        with (
            patch.object(visual, "click_game", return_value=True) as click,
            patch.object(visual, "scroll_game", return_value=True) as wheel,
        ):
            self.assertEqual(
                visual.execute_action(1, FREE_ACTIONS, codes, 7, area, None, output),
                "ACCELERATE+LEFT",
            )
            self.assertEqual(output.pressed, {87, 65})
            key_action = FREE_ACTIONS.index(("key", "SHIFT"))
            self.assertEqual(
                visual.execute_action(key_action, FREE_ACTIONS, codes, 7, area, None, output),
                "SHIFT",
            )
            click_action = FREE_ACTIONS.index(("click", "left", 0.2, 0.22))
            self.assertEqual(
                visual.execute_action(click_action, FREE_ACTIONS, codes, 7, area, None, output),
                "LEFT CLICK",
            )
            click.assert_called_once_with(7, area, (20, 11), None, "left")
            for amount, label in ((120, "WHEEL UP"), (-120, "WHEEL DOWN")):
                action = FREE_ACTIONS.index(("wheel", amount, 0.5, 0.5))
                self.assertEqual(
                    visual.execute_action(action, FREE_ACTIONS, codes, 7, area, None, output), label
                )
            self.assertEqual(wheel.call_count, 2)

    def test_reward_proxy_preserves_road_and_stuck_incentives(self):
        state = np.zeros(2, dtype=np.float32)
        self.assertAlmostEqual(visual.transition_reward(state, state, True, 0, False), 0.6)
        self.assertEqual(visual.transition_reward(state, state, True, None, False), -0.25)
        self.assertEqual(visual.transition_reward(state, state, False, 0, False), -0.2)
        self.assertEqual(visual.transition_reward(state, state, True, 0, True), -2)


class ImitationRuntimeTests(unittest.TestCase):
    def test_self_training_loop_pauses_on_physical_takeover(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            simulated_driver(imitation, directory, takeover_at=4) as run,
        ):
            imitation.drive(self_training=True)
            self.assertTrue(run.output.commands)
            self.assertFalse(run.output.pressed)
            run.keys.close.assert_called_once()

    def test_imitation_model_rejects_unfamiliar_views(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            simulated_driver(imitation, directory) as run,
        ):
            model = run.root / "model.npz"
            np.savez(model, x=np.zeros((7, 4)), y=np.zeros(7))
            with (
                patch.object(imitation, "MODEL", model),
                patch.object(imitation, "features", return_value=np.zeros(4)),
            ):
                imitation.drive()
            self.assertEqual(run.output.commands, [])
            run.keys.close.assert_called_once()

    def test_self_training_missing_speed_or_road_pauses(self):
        for road_lost in (False, True):
            with (
                self.subTest(road_lost=road_lost),
                tempfile.TemporaryDirectory() as directory,
                simulated_driver(imitation, directory, motion=None) as run,
            ):
                with patch.object(
                    imitation,
                    "road_center",
                    return_value=None if road_lost else (80, [(60, 80), (40, 80)]),
                ):
                    imitation.drive(self_training=True)
                self.assertFalse(run.output.pressed)
                run.keys.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
