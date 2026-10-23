"""The installed CLI exposes one RL-first command set without desktop side effects."""

from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from roblinforce import calibration, cli, imitation, visual


@pytest.mark.parametrize(
    "mode, module, operation, arguments",
    [
        ("train", visual, "ai_drive", {"free_exploration": False, "evaluation": False}),
        ("evaluate", visual, "ai_drive", {"free_exploration": False, "evaluation": True}),
        ("explore", visual, "ai_drive", {"free_exploration": True, "evaluation": False}),
        ("evaluate-explore", visual, "ai_drive", {"free_exploration": True, "evaluation": True}),
        ("calibrate", calibration, "run", {}),
        ("record", imitation, "record", {}),
        ("train-imitation", imitation, "train", {}),
        ("drive-imitation", imitation, "drive", {"self_training": False}),
        ("train-steering", imitation, "drive", {"self_training": True}),
    ],
)
def test_mode_dispatch(monkeypatch, mode, module, operation, arguments):
    workflow = Mock()
    monkeypatch.setattr(module, operation, workflow)
    monkeypatch.setattr(sys, "argv", ["roblinforce", mode])
    cli.main()
    workflow.assert_called_once_with(**arguments)


def test_cli_requires_explicit_command(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["roblinforce"])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2


def test_cli_reports_runtime_failure_without_traceback(monkeypatch):
    monkeypatch.setattr(imitation, "train", Mock(side_effect=RuntimeError("No recordings")))
    monkeypatch.setattr(sys, "argv", ["roblinforce", "train-imitation"])
    with pytest.raises(SystemExit, match="No recordings"):
        cli.main()


@pytest.mark.parametrize("entry", ["module", "console"])
def test_installed_help_works_from_another_directory(tmp_path, entry):
    command = (
        [sys.executable, "-m", "roblinforce"]
        if entry == "module"
        else [
            str(
                Path(sys.executable).with_name(
                    "roblinforce.exe" if sys.platform == "win32" else "roblinforce"
                )
            )
        ]
    )
    completed = subprocess.run(
        [*command, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "evaluate-explore" in completed.stdout
    assert "train-imitation" in completed.stdout
    assert "self_train_auto" not in completed.stdout
