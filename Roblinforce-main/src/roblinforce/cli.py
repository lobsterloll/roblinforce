"""One command interface for reinforcement learning and auxiliary experiments."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen-only Roblox reinforcement learning")
    parser.add_argument(
        "mode",
        choices=(
            "train",
            "evaluate",
            "explore",
            "evaluate-explore",
            "calibrate",
            "record",
            "train-imitation",
            "drive-imitation",
            "train-steering",
        ),
        help="Visual reinforcement learning is primary; imitation and steering are auxiliary.",
    )
    args = parser.parse_args()
    # Help and offline training must not initialize native desktop APIs.
    from . import calibration, imitation, visual

    try:
        if args.mode == "calibrate":
            calibration.run()
        elif args.mode == "record":
            imitation.record()
        elif args.mode == "train-imitation":
            imitation.train()
        elif args.mode in ("drive-imitation", "train-steering"):
            imitation.drive(self_training=args.mode == "train-steering")
        else:
            visual.ai_drive(
                free_exploration=args.mode in ("explore", "evaluate-explore"),
                evaluation=args.mode in ("evaluate", "evaluate-explore"),
            )
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        raise SystemExit(str(error)) from error
