"""Evaluation evidence tests verify checkpoint immutability and accounting."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from roblinforce.evaluation import EvaluationLog


class EvaluationTests(unittest.TestCase):
    def test_evidence_preserves_checkpoint_and_summarizes_each_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "agent.npz"
            checkpoint.write_bytes(b"unchanged checkpoint")
            frame = np.zeros((100, 160, 3), dtype=np.uint8)
            with patch("roblinforce.evaluation.DATA", root):
                log = EvaluationLog(checkpoint, (("accelerate",),))
                log.record_action(0, "greedy", True, True)
                log.record_step(0, frame, True, 0.1, 0.5, False, 0.25)
                log.record_step(0, frame, True, None, -0.25, False, 0.4)
                log.record_step(0, frame, False, None, -0.2, False, 0.6)
                log.record_step(0, frame, None, None, -3, True, 1)
                log.close(2)
            self.assertEqual(checkpoint.read_bytes(), b"unchanged checkpoint")
            summary = json.loads((log.directory / "summary.json").read_text())
            self.assertEqual(
                summary["checkpoint_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            )
            self.assertEqual(summary["steps"], 4)
            self.assertEqual(summary["actions"], 1)
            self.assertEqual(summary["detected_road_moving_seconds"], 0.25)
            self.assertEqual(summary["moving_without_aligned_road_seconds"], 0.4)
            self.assertEqual(summary["stopped_seconds"], 0.6)
            self.assertEqual(summary["lost_car_events"], 1)
            self.assertEqual(summary["recovery_attempts"], 2)
            with (log.directory / "steps.csv").open() as stream:
                steps = list(csv.DictReader(stream))
            self.assertEqual(len(steps), 4)
            self.assertEqual(steps[0]["action_id"], "1")
            self.assertEqual(len(list((log.directory / "frames").glob("*.jpg"))), 4)

    def test_failed_frame_write_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "agent.npz"
            checkpoint.write_bytes(b"checkpoint")
            with patch("roblinforce.evaluation.DATA", root):
                log = EvaluationLog(checkpoint, (("accelerate",),))
                try:
                    with patch("roblinforce.evaluation.cv2.imwrite", return_value=False):
                        with self.assertRaisesRegex(OSError, "evaluation frame"):
                            log.record_step(
                                0, np.zeros((10, 10, 3), dtype=np.uint8), False, None, 0, False, 1
                            )
                finally:
                    log.close(0)


if __name__ == "__main__":
    unittest.main()
