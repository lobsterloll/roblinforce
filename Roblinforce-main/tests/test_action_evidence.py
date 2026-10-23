"""An OS input failure must leave evidence of the attempted evaluation action."""

from pathlib import Path
import random
from unittest.mock import Mock

import numpy as np
import pytest

from roblinforce.config import AI_ACTIONS
from roblinforce.learning import VisualLearner
from roblinforce import visual


def test_attempted_action_is_logged_before_input_failure(monkeypatch, tmp_path):
    learner = VisualLearner.load(tmp_path / "agent.npz", 2, len(AI_ACTIONS))
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    state = visual.DrivingState(learner, 42, frame, active=True)
    evidence = Mock()
    context = visual.DrivingContext(
        AI_ACTIONS,
        Path("unused.npz"),
        {},
        {},
        Mock(),
        Mock(),
        random.Random(7),
        evidence,
        False,
        True,
    )
    monkeypatch.setattr(visual, "visual_state", lambda *args: np.zeros(2))
    monkeypatch.setattr(visual, "select_action", lambda *args: (0, "greedy"))

    def failed_input(*args):
        evidence.record_action.assert_called_once_with(0, "greedy", True, True)
        raise OSError("input rejected")

    monkeypatch.setattr(visual, "execute_action", failed_input)
    with pytest.raises(OSError, match="input rejected"):
        visual.known_motion(context, state, None, None, True, True, True, 1.0, {})
    evidence.record_step.assert_not_called()
