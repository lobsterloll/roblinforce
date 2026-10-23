"""Evaluation evidence kept separate from policy updates and live controls."""

import csv
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .config import DATA


@dataclass(frozen=True)
class EvaluationTotals:
    steps: int = 0
    actions: int = 0
    road_seconds: float = 0.0
    unaligned_seconds: float = 0.0
    stopped_seconds: float = 0.0
    lost_events: int = 0


class EvaluationLog:
    """Write action/transition CSVs, sampled frames, and checkpoint identity."""

    def __init__(self, checkpoint: Path, actions: tuple):
        self.checkpoint = checkpoint
        self.checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.actions = actions
        self.totals = EvaluationTotals()
        self.directory = DATA / "evaluations" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        (self.directory / "frames").mkdir(parents=True)
        self.step_file = (self.directory / "steps.csv").open("w", newline="", encoding="utf-8")
        try:
            self.action_file = (self.directory / "actions.csv").open(
                "w", newline="", encoding="utf-8"
            )
        except OSError:
            self.step_file.close()
            raise
        self.step_writer = csv.writer(self.step_file)
        self.step_writer.writerow(
            (
                "action_id",
                "time",
                "frame",
                "action",
                "moving",
                "road_alignment",
                "reward_proxy",
                "stuck",
                "seconds",
            )
        )
        self.action_writer = csv.writer(self.action_file)
        self.action_writer.writerow(
            ("action_id", "time", "action", "selection", "speed_hud_visible", "speed_hud_moving")
        )

    def record_action(
        self, action: int, selection: str, hud_visible: bool, hud_motion: bool | None
    ) -> None:
        self.totals = replace(self.totals, actions=self.totals.actions + 1)
        self.action_writer.writerow(
            (
                self.totals.actions,
                round(time.time(), 3),
                repr(self.actions[action]),
                selection,
                hud_visible,
                hud_motion,
            )
        )
        self.action_file.flush()

    def record_step(
        self,
        action: int,
        frame: np.ndarray,
        moving: bool | None,
        alignment: float | None,
        reward: float,
        stuck: bool,
        duration: float,
    ) -> None:
        totals = self.totals
        changes = {"steps": totals.steps + 1}
        if moving and alignment is not None and alignment <= 0.25:
            changes["road_seconds"] = totals.road_seconds + duration
        elif moving:
            changes["unaligned_seconds"] = totals.unaligned_seconds + duration
        elif moving is False:
            changes["stopped_seconds"] = totals.stopped_seconds + duration
        else:
            changes["lost_events"] = totals.lost_events + 1
        self.totals = replace(totals, **changes)
        image_name = f"{self.totals.steps:06d}.jpg"
        image_width = 640
        image_height = round(frame.shape[0] * image_width / frame.shape[1])
        image = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(
            str(self.directory / "frames" / image_name), image, [cv2.IMWRITE_JPEG_QUALITY, 65]
        ):
            raise OSError("Could not save an evaluation frame.")
        self.step_writer.writerow(
            (
                self.totals.actions,
                round(time.time(), 3),
                image_name,
                repr(self.actions[action]),
                moving,
                alignment,
                reward,
                stuck,
                round(duration, 3),
            )
        )
        self.step_file.flush()

    def close(self, recovery_attempts: int) -> None:
        self.action_file.close()
        self.step_file.close()
        totals = self.totals
        summary = {
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": self.checkpoint_hash,
            "steps": totals.steps,
            "actions": totals.actions,
            "detected_road_moving_seconds": round(totals.road_seconds, 2),
            "moving_without_aligned_road_seconds": round(totals.unaligned_seconds, 2),
            "stopped_seconds": round(totals.stopped_seconds, 2),
            "lost_car_events": totals.lost_events,
            "recovery_attempts": recovery_attempts,
        }
        (self.directory / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"Evaluation saved to {self.directory}")
