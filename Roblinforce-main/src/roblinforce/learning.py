"""Offline steering mathematics and visual policy checkpoints."""

from dataclasses import dataclass, replace
from pathlib import Path
import random

import numpy as np

from .config import STEERING_POLICY, VISUAL_POLICY


def predict(vector: np.ndarray, model_x: np.ndarray, model_y: np.ndarray) -> tuple[int, float, int]:
    similarities = model_x @ vector
    nearest = np.argpartition(similarities, -7)[-7:]
    labels = model_y[nearest]
    votes = np.bincount(labels.astype(np.int32) + 1, minlength=3)
    direction = int(np.argmax(votes)) - 1
    confidence = float(np.max(similarities[nearest]))
    agreement = int(np.max(votes))
    return direction, confidence, agreement


def road_state(
    road: tuple, width: int, car_fraction: float
) -> tuple[tuple[int, int], float, int, float]:
    target, centers = road
    error = (target - width * car_fraction) / (width / 2)
    heading = (centers[-1][1] - centers[0][1]) / width
    state = (
        int(np.digitize(error, (-0.35, -0.22, -0.12, -0.04, 0.04, 0.12, 0.22, 0.35))),
        int(np.digitize(heading, (-0.12, -0.04, 0.04, 0.12))),
    )
    correction = 0.9 * error + 0.25 * heading
    baseline = 0 if abs(correction) < 0.04 else (1 if correction > 0 else -1)
    score = abs(error) + 0.5 * abs(heading)
    return state, score, baseline, error


def update_policy(
    q: np.ndarray,
    visits: np.ndarray,
    state: tuple[int, int],
    action: int,
    reward: float,
    next_state: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a tabular Q update without changing the caller's snapshot."""
    index = state + (action + 1,)
    updated_visits = visits.copy()
    updated_visits[index] += 1
    future = 0.8 * float(np.max(q[next_state])) if next_state is not None else 0.0
    updated_q = q.copy()
    updated_q[index] += 0.2 * (reward + future - q[index])
    return updated_q, updated_visits


def save_policy(q: np.ndarray, visits: np.ndarray) -> None:
    STEERING_POLICY.parent.mkdir(parents=True, exist_ok=True)
    temporary = STEERING_POLICY.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, q=q, visits=visits)
    temporary.replace(STEERING_POLICY)


def visual_values(states: np.ndarray, weights: tuple) -> np.ndarray:
    hidden = np.maximum(states @ weights[0] + weights[1], 0)
    return hidden @ weights[2] + weights[3]


def train_visual(weights: tuple, target_weights: tuple, replay: tuple, rng: random.Random) -> tuple:
    """Apply a clipped DQN minibatch update to a new weight snapshot."""
    batch = rng.sample(replay, 32)
    states = np.stack([item[0] for item in batch])
    actions = np.asarray([item[1] for item in batch], dtype=np.int32)
    rewards = np.asarray([item[2] for item in batch], dtype=np.float32)
    next_states = np.stack([item[3] for item in batch])
    done = np.asarray([item[4] for item in batch], dtype=np.float32)

    hidden_input = states @ weights[0] + weights[1]
    hidden = np.maximum(hidden_input, 0)
    values = hidden @ weights[2] + weights[3]
    targets = rewards + 0.95 * (1 - done) * np.max(
        visual_values(next_states, target_weights), axis=1
    )
    error = np.clip(values[np.arange(32), actions] - targets, -1, 1)
    output_gradient = np.zeros_like(values)
    output_gradient[np.arange(32), actions] = error / 32
    hidden_gradient = (output_gradient @ weights[2].T) * (hidden_input > 0)
    gradients = (
        states.T @ hidden_gradient,
        np.sum(hidden_gradient, axis=0),
        hidden.T @ output_gradient,
        np.sum(output_gradient, axis=0),
    )
    return tuple(
        weight - 0.005 * np.clip(gradient, -1, 1) for weight, gradient in zip(weights, gradients)
    )


def save_visual(weights: tuple, steps: int, best_streak: float, path: Path = VISUAL_POLICY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output,
            w1=weights[0],
            b1=weights[1],
            w2=weights[2],
            b2=weights[3],
            steps=steps,
            best_streak=best_streak,
        )
    temporary.replace(path)


@dataclass(frozen=True)
class VisualLearner:
    """Policy snapshots and bounded replay with the original checkpoint format."""

    weights: tuple[np.ndarray, ...]
    target_weights: tuple[np.ndarray, ...]
    replay: tuple[tuple, ...] = ()
    steps: int = 0
    best_streak: float = 0.0

    @classmethod
    def load(cls, path: Path, input_size: int, action_count: int) -> "VisualLearner":
        initializer = np.random.default_rng()
        shapes = ((input_size, 64), (64,), (64, action_count), (action_count,))
        weights = (
            (initializer.standard_normal(shapes[0]) * np.sqrt(2 / input_size)).astype(np.float32),
            np.zeros(shapes[1], dtype=np.float32),
            (initializer.standard_normal(shapes[2]) * np.sqrt(2 / 64)).astype(np.float32),
            np.zeros(shapes[3], dtype=np.float32),
        )
        steps, best_streak = 0, 0.0
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                weights = tuple(saved[name].astype(np.float32) for name in ("w1", "b1", "w2", "b2"))
                steps = int(saved["steps"])
                best_streak = float(saved["best_streak"])
            if any(weight.shape != shape for weight, shape in zip(weights, shapes)):
                raise RuntimeError("The saved visual policy has an invalid shape.")
        return cls(weights, weights, steps=steps, best_streak=best_streak)

    def observe(self, transition: tuple, rng: random.Random, path: Path) -> "VisualLearner":
        replay = (*self.replay[-1999:], transition)
        weights = (
            train_visual(self.weights, self.target_weights, replay, rng)
            if len(replay) >= 32
            else self.weights
        )
        steps = self.steps + 1
        # Fixed targets keep one minibatch from chasing its own freshly updated estimates.
        target_weights = weights if steps % 200 == 0 else self.target_weights
        updated = replace(
            self, weights=weights, target_weights=target_weights, replay=replay, steps=steps
        )
        if steps % 100 == 0:
            updated.save(path)
        return updated

    def save(self, path: Path) -> None:
        save_visual(self.weights, self.steps, self.best_streak, path)
