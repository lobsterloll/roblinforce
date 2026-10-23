# Development guide

## Project purpose and boundaries

Roblinforce is a supervised, screen-only Roblox Driving Empire prototype. Visual reinforcement learning is the primary workflow; imitation learning and a steering table are retained as auxiliary modes in one package under `src/roblinforce/`. It reads screenshots and sends desktop input; it must never read or modify Roblox memory.

Windows and macOS are supported desktop targets. Linux is unsupported. No routing, reliable intersection handling, collision detection, or unattended-driving guarantee exists. Do not describe screenshot-derived rewards as verified driving success. LangChain/LangGraph are not dependencies and should not be introduced without a concrete requirement.

## Documentation ownership

- `README.md`: installation, source build checks, launch commands, learning/evaluation procedures, major libraries, controls, troubleshooting, and manual acceptance checks for users.
- `ARCHITECTURE.md`: actual modules, data flow, OS boundaries, model formats, and technical limitations.
- `AGENTS.md`: instructions for contributors and coding agents. Link to user instructions instead of duplicating them here.
- `skills/`: canonical local workflow instructions; read `skills/self-documenting-code/SKILL.md` before code changes.

Use Markdown for documentation and English for primary documents, code comments, and docstrings. Name Korean translations `NAME-KO.md`. Update existing documents rather than adding overlapping top-level guides. Do not present a proposal, an old executable, or an unperformed live test as implemented/verified behavior.

## Repository navigation

Application code belongs only in `src/roblinforce/`. Do not reintroduce version-specific trees, duplicate entry points, or a root compatibility facade.

```text
roblinforce/
├── src/roblinforce/          # Single maintained application package
│   ├── __init__.py          # Package identity
│   ├── __main__.py          # python -m roblinforce
│   ├── cli.py               # RL-first dispatch and auxiliary commands
│   ├── config.py            # Paths and stable action ordering
│   ├── settings.py          # Validated settings/key bindings
│   ├── calibration.py       # Preview-only road-color calibration
│   ├── desktop.py           # Shared focus/input/capture contract
│   ├── desktop_windows.py   # Win32 integration
│   ├── desktop_macos.py     # Quartz/AppKit integration
│   ├── road.py              # Conservative road detection and overlays
│   ├── perception.py        # Observations, speed HUD, vehicle-menu geometry
│   ├── learning.py          # Visual Q-network, replay, KNN, table, persistence
│   ├── visual.py            # Primary RL training/evaluation control loop
│   ├── imitation.py         # Auxiliary recording, imitation, tabular driving
│   ├── recovery.py          # Interruptible car selection and respawn
│   ├── evaluation.py        # Attempted/completed actions, frames, summaries
│   └── assets/speed-zero.png # Runtime HUD reference
├── tests/                   # Pure, fake-OS, CLI, and simulated runtime tests
├── models/                  # Local learned checkpoints (ignored)
├── data/                    # Local recordings/evaluation evidence (ignored)
├── work/                    # Reference and debug images
├── artifacts/legacy/        # Preserved obsolete builds (ignored)
├── settings.json            # Local calibration and key bindings
├── pyproject.toml           # Dependencies, entry point, test/lint settings
├── requirements*.txt        # Editable-install convenience wrappers
├── README.md                # User installation, operation, and verification
├── ARCHITECTURE.md          # Module contracts, learning, and OS boundaries
├── AGENTS.md                # Contributor and AI-agent instructions
├── skills/self-documenting-code/SKILL.md
└── .trail/                  # Generated repository context graph
```

Preserve configuration, checkpoints, recordings, and reference images unless a requested change explicitly requires modifying them. The repository may contain pre-existing untracked files: inspect status before editing and never discard them. Generated folders may be absent in a fresh checkout; code creates output directories when needed.

### Route a change to its owner

| Intended change | Start here |
| --- | --- |
| Command or mode name | `cli.py`, `tests/test_cli.py`, README command table |
| Reward, action scheduling, training/evaluation lifecycle | `visual.py`, `tests/test_runtime.py` |
| Model update, checkpoint schema, replay | `learning.py`, `tests/test_learning.py` |
| Observation or HUD/road/menu detection | `perception.py`, `road.py`, corresponding tests |
| Calibration or configuration | `calibration.py`, `settings.py`, their tests |
| Native capture, focus, input, permissions | `desktop.py`, platform backend, fake desktop tests |
| Recovery interruption | `recovery.py`, `tests/test_runtime.py` |
| Evidence format | `evaluation.py`, `tests/test_evaluation.py` |
| Auxiliary learning | `imitation.py`, `tests/test_io.py`, `tests/test_runtime.py` |

Major libraries are NumPy (arrays/learning/checkpoints), OpenCV (vision/previews), MSS (capture), and PyObjC/Quartz (macOS native integration). Windows uses standard-library `ctypes`. Read the README library table for purpose and upstream links; keep version ranges in `pyproject.toml` instead of duplicating them. No deep-learning or agent framework is required by the current implementation.

### Vibe Coding task brief

Before coding, turn the request into an observable outcome. A useful short brief is:

```text
Desired behavior: what the operator should observe
Current behavior: evidence from code, logs, or a failing test
Scope: owning modules and contracts that must stay stable
Acceptance: automated checks and any separate supervised game check
```

Use the task table and Trail graph to locate existing behavior. For a defect, reproduce it with synthetic input or fake OS calls before changing code. For a feature, identify the command, model/data contract, and interruption behavior it touches. Choose the smallest coherent edit; keep explanations of non-obvious constraints in English comments. Show test evidence and remaining live-check limitations in the completion report. Do not infer success from a plausible-looking preview or a training counter.

<!-- trail:start -->

## Trail — repo context graph

This repo is indexed in `.trail/cache/`: small linked markdown nodes that explain each
system and carry exact file:line spans, kept in sync with the code through git.

For ANY task here — understanding how something works, finding where code lives,
or scoping a change — get context from the graph before grepping or opening
source files. Re-ask freely (it's cheap) and reuse literal identifiers you
already have (symbol, error string, file name) as the query. New to this repo?
Run `trail map` first — a token-budgeted orientation (dir clusters, hubs,
hotspots), no LLM, no key.

- Run `trail ask "<your question>" --source` → ranked nodes with the relevant
  code spans inlined (each hit's ≤8-line crux by default; `--full` for whole
  definitions when the crux isn't enough). Match the tool to the task shape:
  for understanding or editing, the top node IS the answer — cite its
  `covers:` file:line spans and edit straight from `--source`. For
  exhaustive tasks ("every occurrence / every caller of this pattern"), ranked
  results are top-N, not complete — run `trail grep "<literal>"` instead
  (exhaustive over indexed files, grouped by enclosing symbol), falling back
  to raw `grep -rn` only for unindexed files.
- `trail skeleton <file>` → every definition's signature + span, ~10× cheaper
  than reading the file; use it to skim an API surface.
- `trail callers <symbol>` gives precomputed, exact edges — who calls this.
  Add `--direction out` for what it calls, or `--depth N` to walk
  transitively for the full blast radius. For structural questions, skip
  ranking and use this directly.
- Or browse: `.trail/cache/INDEX.md` lists every node; follow the links.
- Monorepos and folders of multiple repos rank fairly across sub-projects —
  hits carry `[scope/]` labels naming which one they're from. Narrow with
  `trail ask "<task>" --in <scope>/` once you know where you're working.

If a returned span is truncated ("+N more lines"), open the file at that exact
range before finalizing. Only open source files when a node genuinely lacks a
needed detail, and then at the exact file:line the node points to — never
re-read whole files.

After big code changes, refresh the graph with `trail build` (deterministic,
no API key, $0).
<!-- trail:end -->

## Implementation workflow

1. Restate the intended outcome and completion criteria. Inspect the graph, existing behavior, dependencies, and working tree before editing.
2. Plan complex changes. Identify the minimum files and reusable functions. Explain why more than three files must change before a broad edit.
3. Delegate independent domain work when appropriate. Assign file ownership, tell workers they share the checkout, and never revert another worker's changes.
4. For behavior changes and bug fixes, write a meaningful failing test first, implement the smallest fix, and refactor while keeping the test green. Preserve mode names, checkpoint shapes, paths, and keyboard behavior unless a change is intentional and documented.
5. Review the final changes for correctness, security, readability, and accidental behavior changes. Use a separate review agent for substantial changes. Remove redundant helpers, comments, abstractions, dead code, and unrelated cleanup.
6. Run relevant tests and the verification commands below. Report actual results and distinguish simulated desktop tests from manual OS/game acceptance.
7. Update documentation in its existing location. Keep personal temporary notes outside project documentation. Use conventional commit subjects if a commit is requested.

External services are read-only by default. Do not push, publish, merge, send messages to others, change credentials, or start paid/remote jobs without the user's explicit authorization. Local subagents, local reversible edits, and offline tests are permitted within the requested task.

## Self-documenting Python

- Follow PEP 8, public type annotations, and concise docstrings that explain contracts.
- Prefer names that reveal domain intent, named policy thresholds, explicit data flow, semantic Boolean conditions, and guard clauses.
- Extract a function/module only for a coherent responsibility, reuse, or testability. Avoid factories, managers, wrappers, speculative extension points, and excessive file splitting.
- Aim for focused functions under 50 lines and modules around 200–400 lines. Keep modules below 800 lines; explain exceptions for orchestration that cannot be split clearly.
- Use one meaningful operation per statement. Do not compress unrelated transformations into a clever expression or add intermediate variables that convey no meaning.
- Comments explain why: OS limitations, event filtering, perception assumptions, coordinate conventions, safety constraints, and learning formulas. Do not narrate obvious code.
- Prefer immutable configuration/observations and newly returned values. In-place NumPy model updates, image drawing, and OS/event state may be necessary; keep such mutation explicit and local to the responsible component.
- Handle realistic failures at the appropriate boundary. User-facing errors must explain what failed. Never silently suppress input, capture, listener, or persistence failures.

## Desktop and input safety

- Keep native OS imports behind the desktop boundary. Importing perception, loading a checkpoint, training offline, or asking for CLI help must not create a hook, request permission, capture the screen, or emit input.
- Start driving paused. Confirm target-window focus before each new key/mouse action. Always allow key/button release after focus loss.
- Preserve F8 pause/start, F9 stop where supported, and physical-driving-key takeover. Free exploration currently disables F9; document that exception and preserve physical takeover/preview close.
- Ignore the assistant's synthetic key events when observing physical input. Listener startup errors or listener death must fail closed and release keys.
- Release held keys in `finally` paths, on pause, focus loss, missing windows, uncertain road tracking in road-following modes, and unexpected failures.
- Keep capture and click coordinates consistent, including Retina scaling. Do not inject Windows API calls into shared learning/perception code.
- Check macOS permissions explicitly and explain the required System Settings permissions. Do not change system privacy settings programmatically.
- Never run real desktop capture or key/mouse injection in automated tests. Live game acceptance requires supervision and explicit intent to exercise the game.

## Configuration, models, and data integrity

- Never hardcode or commit credentials. No network/API secrets are needed for the current application.
- Validate settings at load boundaries: supported/distinct key names, numeric ranges, finite values, and expected LAB color shape. Keep file locations explicit and independent of the current directory.
- Load NumPy archives without pickle; check model shapes before inference/training. Never introduce unsafe `torch.load` or YAML loading. If YAML is ever required, use `yaml.safe_load`.
- Treat external paths as untrusted. Normalize and constrain them to their intended location before reading/writing; do not add arbitrary path controls without a requirement.
- Preserve checkpoints and recordings during refactors. Evaluation must not train, overwrite, or save its input checkpoint; verify the checkpoint hash remains unchanged.
- Keep action evidence and completed-step evidence distinct. A planned/attempted action is not proof that input succeeded or that the car advanced.

## Verification

Use the interpreter inside the project virtual environment: `.venv/bin/python` on macOS, `.\.venv\Scripts\python.exe` on Windows. From the repository root:

```sh
python -m compileall -q src
python -m ruff check src tests
python -m pytest --cov --cov-report=term-missing
python -m roblinforce --help
roblinforce --help
```

Install the checkout with `python -m pip install -e ".[dev]"` (or `requirements-dev.txt`). The editable install is required for installed-entry-point tests; standalone wheel runtime deployment is unsupported. The coverage threshold is **80%**. Cover pure perception/model functions, persistence, fake OS integration, and simulated critical control flows. Test failures should lead to implementation fixes unless the assertion itself is wrong. Do not lower the gate or hide production modules solely to make the number pass.

Run the README's manual acceptance checks on Windows and macOS before claiming live compatibility. No automated test can establish permission grants, Roblox input acceptance, actual car recovery, or useful learning. Before any requested commit, inspect the diff, check for secrets and unsafe serialization, and audit runtime dependencies when tooling/network access permits. Never claim checks that were not run.
