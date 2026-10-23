# Roblinforce

A supervised, screen-only **reinforcement learning** prototype for Roblox Driving Empire on **Windows and macOS**. One Python package captures the visible game window, estimates movement and road alignment, and learns keyboard actions. It never reads Roblox memory or game coordinates.

Visual reinforcement learning is the primary workflow. Manual imitation learning and tabular steering remain auxiliary experiments in the same package. All driving modes start paused. The policies cannot choose a destination, reliably handle intersections, detect collisions, or prove route completion. Supervise live runs.

## Requirements and installation

- Python **3.11 or newer**, a desktop session, and the GUI `opencv-python` package.
- A visible Roblox Driving Empire window and a steady rear camera. Run one driver process at a time.
- Windows: Roblox and Python at the same privilege level.
- macOS: Screen Recording, Accessibility, and Input Monitoring permission for the application launching Python.

Linux desktop control is unsupported. Tests and offline imitation training do not need a running game. Use an **editable installation from this checkout**: writable settings, recordings, and models are anchored to the repository, independent of your shell's current directory.

### Windows (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m roblinforce --help
.\.venv\Scripts\python.exe -m roblinforce calibrate
```

### macOS (Terminal)

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m roblinforce --help
.venv/bin/python -m roblinforce calibrate
```

Run installation commands from the repository root. `requirements.txt` is an editable-install wrapper for the same dependencies; `pyproject.toml` is their source of truth. After activating the virtual environment, `roblinforce MODE` and `python -m roblinforce MODE` are equivalent. Examples below use `python`; substitute your virtual environment's interpreter when it is not activated.

On macOS, grant **Screen Recording** (also called **Screen & System Audio Recording** on some versions), **Accessibility**, and **Input Monitoring** in System Settings → Privacy & Security to Terminal, your IDE, or whichever application launches Python. Restart that application after grants. The shared startup check requires all three permissions, including for calibration. Use **Fn+F8/F9** if function keys default to media controls.

The macOS adapter normalizes Retina captures to logical window coordinates. Its capture may include window decorations, unlike the Windows client-area capture. Recalibrate after changing platform, camera, or window size. Native gameplay acceptance on each OS is still required.

## Calibrate before driving

1. Open Driving Empire, enter a car, and choose a steady rear camera with **V**. Begin on a straight, empty road.
2. Run `python -m roblinforce calibrate`. This preview does not inject input or start a keyboard listener.
3. Focus Roblox to refresh the frame, then click a clean gray asphalt patch ahead of the car in the preview. Saturated scenery is rejected. Accepted color is saved in `settings.json`.
4. Check that green road points follow the asphalt and the yellow reference matches the car's forward position. Press Q or close the preview.
5. Launch a driving mode, focus Roblox, and deliberately press F8 to start.

Driving commands reject a missing road calibration before accessing the desktop.

## Commands

| Command | Purpose |
| --- | --- |
| `train` | Train the primary visual road-goal policy in the live game. |
| `evaluate` | Evaluate its frozen checkpoint and save evidence. |
| `explore` | Train a separate policy with wider game keys, clicks, and scrolling. |
| `evaluate-explore` | Evaluate the frozen explorer checkpoint. |
| `calibrate` | Preview and save asphalt color without controlling the game. |
| `record` | Record manual steering examples; F9 finishes and saves. |
| `train-imitation` | Build the auxiliary imitation model offline. |
| `drive-imitation` | Drive using that imitation model and a road-confidence check. |
| `train-steering` | Train the auxiliary tabular steering policy from road alignment. |

### Primary reinforcement learning

```sh
python -m roblinforce train
python -m roblinforce evaluate
```

`train` observes a reduced grayscale screenshot plus a movement estimate. It chooses acceleration, acceleration+left/right, left/right, brake, or handbrake using your configured bindings. Road alignment and movement provide reward; stopped, off-road, or missing-car evidence incurs penalties. Consecutive acceleration actions can hold the accelerator continuously. `throttle_duty` does not cap visual modes.

The checkpoint is `models/road_goal_agent.npz`. The preview shows detected road points and current/best apparent road-driving streak. These are screen-derived estimates, not verified progress or collision avoidance.

After prolonged apparent zero speed (normally 20 seconds), recovery attempts to leave/reselect the car using F/T and visible vehicle-menu geometry. Missing HUD evidence can trigger an earlier attempt. Prolonged missing-car evidence can trigger Esc → R → Enter respawn. Recovery checks focus and physical interruption between steps. Changed menus, camera, resolution, or platform can break these heuristics; inspect failed attempts before resuming.

### Wider exploration

```sh
python -m roblinforce explore
python -m roblinforce evaluate-explore
```

The separate checkpoint is `models/game_explorer_agent.npz`. Its 72 actions include driving, selected letters/numbers, Space/Shift/arrows, clicks, and scrolling above the bottom quarter of the game view. When the speed HUD is visible, it starts a stopped car with acceleration and restricts moving choices to acceleration/acceleration+left/right. Outside the car it can use the wider set. Random exploration during training remains at least 25%.

The action list excludes Esc, L, Alt, Command/Windows, F8/F9, and window controls. Outside-car image novelty can earn a small reward even when the behavior does not help driving. **F9 and Q-to-quit are disabled in both explorer modes.** Use F8, physical driving keys/Esc, or close the preview to interrupt.

### Auxiliary learning

```sh
python -m roblinforce record
python -m roblinforce train-imitation
python -m roblinforce drive-imitation
python -m roblinforce train-steering
```

During `record`, focus Roblox and drive manually through left, straight, and right examples. Roughly five times per second while you physically hold the accelerator, the recorder stores a normalized HOG feature and steering label, not a full screenshot. F9 finishes and saves `data/drive-*.npz`. Record at least **20 examples of each steering class**.

`train-imitation` samples up to 1,500 examples per class across recordings and writes `models/model.npz`. Retrain to include new sessions. `drive-imitation` uses nearest-example voting and pauses on low similarity, uncertain votes, or insufficient road confidence.

`train-steering` learns left/straight/right corrections from road alignment without manual recordings and saves `models/steering_q.npz`. It pauses on road/focus loss, manual takeover, excessive alignment error, or prolonged apparent zero speed. Inspect/reset the car before pressing F8 again. Neither auxiliary mode learns routes or destinations.

## Controls and focus

Default bindings are shown; `settings.json` can change driving/toggle/stop keys.

| Mode | F8 | F9 | Interruption / quit |
| --- | --- | --- | --- |
| `calibrate` | Unused | Unused | Q or preview close exits. |
| `record` | Unused | Finish and save | Watches manual keys without driving. |
| `drive-imitation`, `train-steering` | Start/pause | Release keys and pause | Physical driving key/Esc pauses; Q or preview close exits. |
| `train`, `evaluate` | Start/pause | Release keys and pause | Physical driving key/Esc pauses; Q or preview close exits. |
| `explore`, `evaluate-explore` | Start/pause | Disabled | Physical driving key/Esc pauses; preview close exits. |

MANUAL TAKEOVER means paused; recovery is also paused. Press F8 deliberately to resume. F9 releases assistant input but does not brake a coasting car. Use your own brake/handbrake. A process freeze or force termination can prevent cleanup.

New input requires Roblox focus. Visual modes release input on focus/window loss and can resume when Roblox returns while still active. Imitation/tabular modes require F8 again. Capture, input, and listener failures should release held keys and surface an error.

## Evaluate and inspect evidence

Evaluation requires the corresponding checkpoint, uses greedy choices, and never updates or saves its weights. Explorer evaluation retains the in-car action restrictions. Evaluation still controls the game and may attempt recovery.

Use the same road, car, camera, window size, and comparable durations for repeated comparisons. Each run creates `data/evaluations/<timestamp>/`:

| Output | Meaning |
| --- | --- |
| `actions.csv` | Attempted actions and selection reasons, logged when selected. |
| `steps.csv` | Completed observations, action ID, duration, movement/alignment, and reward proxy. |
| `frames/*.jpg` | Outcome images for visual review. |
| `summary.json` | Checkpoint path/hash, counts, estimated aligned-road time, unaligned movement, stopped time, lost-car events, and recovery attempts. |

Inspect frames with both CSVs: selecting an action does not prove it completed. Metrics do not measure true distance or collisions. Check the checkpoint hash before and after evaluation:

```sh
# macOS; repeat after evaluation.
shasum -a 256 models/road_goal_agent.npz
```

```powershell
# Windows; repeat after evaluation.
Get-FileHash models\road_goal_agent.npz -Algorithm SHA256
```

## Settings and local files

| Setting | Meaning |
| --- | --- |
| `window_title` | Target Roblox window title. Keep it specific. |
| `keys` | Distinct supported acceleration, brake, handbrake, steering, toggle, and stop bindings. |
| `road_sample_lab` | Calibrated OpenCV LAB asphalt color; `null` requires calibration. |
| `color_tolerance` | Color tolerance for the conservative road detector used by auxiliary driving. |
| `car_x_fraction` | Horizontal car anchor, from 0 (left) to 1 (right). |
| `throttle_duty` | Accelerator duty cycle for auxiliary imitation/tabular driving, not visual RL. |

| Location | Contents |
| --- | --- |
| `src/roblinforce/` | The only maintained application source. |
| `src/roblinforce/assets/` | Runtime speed-HUD reference image. |
| `settings.json` | Repository-local configuration and calibration. |
| `models/` | Imitation, steering, road-goal, and explorer checkpoints. |
| `data/` | Manual recordings and evaluation evidence. |
| `work/` | Reference/debug images. |
| `artifacts/legacy/` | Archived obsolete builds and executable; not a current release. |

Back up recordings and models before retraining. Existing checkpoint array names, shapes, and action ordering are preserved. `models/visual_agent.npz` is an older movement-only checkpoint retained as data; active road-goal training uses `road_goal_agent.npz`.

The former versioned entry points have been removed. Use the command table above: `train` now means visual reinforcement learning, while offline imitation training is explicitly `train-imitation`. There is one source package and one user guide.

## Major open-source libraries

| Library | Role in this project |
| --- | --- |
| [NumPy](https://numpy.org/doc/stable/user/whatisnumpy.html) | Image/feature arrays, custom visual Q-network math and updates, replay batches, tabular learning, and NPZ checkpoint persistence. |
| [OpenCV](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) | LAB/HSV conversion, image resizing/morphology, HOG features, HUD/template checks, calibration/driver previews, and evidence images. Install `opencv-python` with GUI support. |
| [MSS](https://python-mss.readthedocs.io/) | Captures the target window's screen rectangle through the shared desktop adapter. |
| [PyObjC / Quartz](https://pyobjc.readthedocs.io/en/latest/) | macOS bridge to Quartz/AppKit for window discovery, permissions, physical key events, and keyboard/mouse output. Installed only on macOS. |

Windows native access uses Python's standard-library `ctypes`. Learning is implemented directly with NumPy; no PyTorch, Gym, Stable-Baselines, LangChain, or LangGraph dependency is required. Development uses pytest, pytest-cov, Ruff, and setuptools; supported dependency ranges are in `pyproject.toml`.

## Build and automated verification

This is a Python source application. Installation registers the package and console command; no native compiler is required. There is no maintained standalone EXE/app release pipeline. Archived executables predate the current source. Use the editable checkout, not a copied package directory or standalone wheel deployment.

From the repository root, install development dependencies and run:

```sh
# macOS
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m compileall -q src
.venv/bin/python -m ruff check src tests
.venv/bin/python -m pytest --cov --cov-report=term-missing
.venv/bin/python -m roblinforce --help
.venv/bin/roblinforce --help
```

```powershell
# Windows
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m compileall -q src
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m pytest --cov --cov-report=term-missing
.\.venv\Scripts\python.exe -m roblinforce --help
.\.venv\Scripts\roblinforce.exe --help
```

`requirements-dev.txt` wraps the same editable development installation. The coverage gate is **80%** across all application modules. Tests use synthetic frames and fake desktop APIs; they do not capture the screen or inject input. Tests include CLI dispatch, checkpoint persistence/evaluation immutability, coordinate conversion, listener failures, focus loss, takeover, and recovery interruption.

## Verify live operation

Run these checks under supervision on each target OS; automated tests do not establish native/game compatibility:

1. With Roblox closed, launch `calibrate`: expect a clear missing-window error. On macOS, confirm missing permissions are reported before proceeding.
2. Open Roblox, calibrate, and verify the captured frame and road overlay. On Retina displays, verify clicked sample alignment after resizing.
3. Launch `train`: expect paused state. Focus Roblox and press F8; observe actions on an empty road. Test F8 pause, F9 stop, and physical steering/brake takeover separately.
4. Switch applications while active: input must be released and never sent to the other application. Check minimized/unavailable-window behavior and the documented visual-mode resumption behavior.
5. Quit through the preview and confirm key release. Verify recovery clicks stay inside the game window; inspect car selection/respawn separately.
6. Record left/straight/right examples, run `train-imitation` offline, then load `drive-imitation`. Inspect `train-steering` separately.
7. Evaluate a backed-up checkpoint, compare its before/after hash, and inspect CSVs and saved frames. Verify explorer interruption controls separately before prolonged use.

No automated test proves useful learning, collision avoidance, or successful live recovery.

## Troubleshooting and documentation

| Symptom | Check |
| --- | --- |
| Module/command not found | Install this checkout with the same virtual environment used to run it. |
| No Roblox window | Open a visible game window and check `window_title`. |
| Permission/listener error on macOS | Grant all three permissions to the launching application and restart it. |
| F8/F9 do nothing | Check Fn/media-key settings, mode-specific F9 behavior, and listener errors; close other drivers. |
| Black/stale preview | Check capture permission, visibility, minimization, and game focus. |
| Incorrect road overlay | Recalibrate asphalt; check rear camera, lighting, and car anchor. |
| Model missing | Train the matching visual policy, or record and run `train-imitation`. |
| OpenCV GUI error | Use a desktop session and `opencv-python`, not the headless variant. |

[ARCHITECTURE.md](ARCHITECTURE.md) explains modules, model formats, and technical boundaries. [AGENTS.md](AGENTS.md) describes repository navigation and contributor/AI coding workflow. Documentation is Markdown and English by default; Korean translations use `-KO.md`.
