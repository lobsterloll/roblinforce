"""Checkpoint locations and stable action ordering used by existing agents."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = Path(__file__).resolve().parent / "assets"
MODELS = ROOT / "models"
DATA = ROOT / "data"
MODEL = MODELS / "model.npz"
STEERING_POLICY = MODELS / "steering_q.npz"
VISUAL_POLICY = MODELS / "road_goal_agent.npz"
FREE_POLICY = MODELS / "game_explorer_agent.npz"
AI_ACTIONS = (
    ("accelerate",),
    ("accelerate", "left"),
    ("accelerate", "right"),
    ("left",),
    ("right",),
    ("brake",),
    ("handbrake",),
)
FREE_ACTIONS = (
    AI_ACTIONS
    + tuple(("key", name) for name in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" if name not in "WASDL")
    + (
        ("key", "SPACE"),
        ("key", "SHIFT"),
        ("key", "UP"),
        ("key", "DOWN"),
        ("key", "LEFT"),
        ("key", "RIGHT"),
    )
    + tuple(
        ("click", button, x, y)
        for button in ("left", "right")
        for x in (0.2, 0.4, 0.6, 0.8)
        for y in (0.22, 0.45, 0.68)
    )
    + tuple(("wheel", direction, x, 0.5) for direction in (-120, 120) for x in (0.5, 0.85))
)
