"""Supervised visual learning and deterministic checkpoint evaluation loops."""

from dataclasses import dataclass, replace
from contextlib import ExitStack
import random
import time
from pathlib import Path

import cv2
import mss
import numpy as np

from . import desktop
from .settings import load_settings
from .config import AI_ACTIONS, FREE_ACTIONS, FREE_POLICY, VISUAL_POLICY
from .evaluation import EvaluationLog
from .learning import VisualLearner, road_state, visual_values
from .perception import (
    car_category_icon,
    car_tile,
    scene_moving,
    screen_road,
    speed_hud_visible,
    speed_moving,
    visual_state,
)
from .recovery import click_game, recover_car, respawn_character, scroll_game


PREVIEW_TITLE = "Driving Empire visual AI"


def select_action(
    state: np.ndarray,
    learner: VisualLearner,
    actions: tuple,
    rng: random.Random,
    free_exploration: bool,
    evaluation: bool,
    moving: bool,
    hud_visible: bool,
    hud_motion: bool | None,
) -> tuple[int, str]:
    epsilon = 0.0 if evaluation else max(0.08, 0.40 - 0.32 * learner.steps / 50000)
    if free_exploration and not evaluation:
        epsilon = max(epsilon, 0.25)
    if not moving and not evaluation:
        epsilon = max(epsilon, 0.65)
    in_car = free_exploration and hud_visible
    if in_car and hud_motion is False:
        return 0, "start_car"
    if rng.random() < epsilon:
        return rng.randrange(3 if in_car else len(actions)), "explore"
    values = visual_values(state, learner.weights)
    return int(np.argmax(values[:3] if in_car else values)), "greedy"


def execute_action(action: int, actions: tuple, codes: dict, hwnd, area: dict, keys, output) -> str:
    if action < len(AI_ACTIONS):
        output.set_pressed({codes[name] for name in AI_ACTIONS[action]})
        return "+".join(AI_ACTIONS[action]).upper()
    output.release_all()
    kind, *details = actions[action]
    if kind == "key":
        output.set_pressed({desktop.virtual_key(details[0])})
        return details[0]
    point = (int(area["width"] * details[1]), int(area["height"] * details[2]))
    if kind == "click":
        click_game(hwnd, area, point, keys, details[0])
        return f"{details[0].upper()} CLICK"
    scroll_game(hwnd, area, point, keys, details[0])
    return "WHEEL UP" if details[0] > 0 else "WHEEL DOWN"


def transition_reward(
    state: np.ndarray,
    previous: np.ndarray,
    moving: bool | None,
    alignment: float | None,
    stuck: bool,
) -> float:
    if stuck:
        return -2.0
    is_aligned = moving and alignment is not None and alignment <= 0.25
    if not is_aligned:
        return -0.25 if moving else -0.20
    # Pixel changes are only a small bonus; the principal goal remains aligned motion.
    visual_change = float(np.mean(np.abs(state[:-1] - previous[:-1])))
    return 0.20 + 0.40 * (1 - alignment / 0.25) + min(0.10, 2 * visual_change)


def preview_closed(free_exploration: bool) -> bool:
    key = cv2.waitKey(20) & 0xFF
    quit_pressed = not free_exploration and key == ord("q")
    return quit_pressed or cv2.getWindowProperty(PREVIEW_TITLE, cv2.WND_PROP_VISIBLE) < 1


def show_preview(
    frame: np.ndarray, road: tuple | None, status: str, free_exploration: bool
) -> None:
    scale = min(1.0, 1000 / frame.shape[1])
    preview = cv2.resize(frame, None, fx=scale, fy=scale)
    if road is not None:
        for y, x in road[1]:
            cv2.circle(preview, (int(x * scale), int(y * scale)), 5, (0, 255, 0), -1)
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(preview, status, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    controls = (
        "F8 start/pause | physical driving key takes over | close window to quit"
        if free_exploration
        else "F8 start/pause | F9 stop | physical driving key takes over | Q quit"
    )
    cv2.putText(preview, controls, (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    road_label = "ROAD VISIBLE" if road is not None else "NO ROAD DETECTED"
    color = (0, 220, 0) if road is not None else (0, 0, 255)
    cv2.putText(preview, road_label, (12, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.imshow(PREVIEW_TITLE, preview)


@dataclass(frozen=True)
class DrivingState:
    """One loop snapshot; interrupted observations never leak into a later transition."""

    learner: VisualLearner
    window: int | None
    frame: np.ndarray
    active: bool = False
    status: str = "VISUAL AI PAUSED - focus Roblox, then F8"
    last_state: np.ndarray | None = None
    last_action: int | None = None
    lost_transition: tuple | None = None
    onroad_streak: float = 0.0
    action_since: float = 0.0
    cooldown_until: float = 0.0
    stuck_since: float | None = None
    missing_since: float | None = None
    retry_recovery_at: float = 0.0
    motion_frame: np.ndarray | None = None
    motion_at: float = 0.0
    scene_motion: bool = False
    last_road_at: float | None = None
    offroad_since: float | None = None
    seen_views: frozenset = frozenset()
    recovery_attempts: int = 0

    def clear_tracking(self, **changes) -> "DrivingState":
        return replace(
            self,
            last_state=None,
            lost_transition=None,
            onroad_streak=0.0,
            stuck_since=None,
            missing_since=None,
            motion_frame=None,
            scene_motion=False,
            last_road_at=None,
            offroad_since=None,
            **changes,
        )


@dataclass(frozen=True)
class DrivingContext:
    actions: tuple
    policy_path: Path
    settings: dict
    codes: dict
    keys: object
    output: object
    rng: random.Random
    evidence: EvaluationLog | None
    free_exploration: bool
    evaluation: bool


def poll_controls(context: DrivingContext, state: DrivingState) -> DrivingState:
    keys, output = context.keys, context.output
    keys.check_health()
    for event, status in (
        (keys.stop, "VISUAL AI STOPPED - all keys released"),
        (keys.takeover, "VISUAL AI MANUAL TAKEOVER - F8 to resume"),
    ):
        if event.is_set():
            event.clear()
            keys.toggle.clear()
            output.release_all()
            if not context.evaluation:
                state.learner.save(context.policy_path)
            state = replace(
                state,
                active=False,
                last_state=None,
                lost_transition=None,
                onroad_streak=0.0,
                status=status,
            )
    if keys.toggle.is_set():
        keys.toggle.clear()
        active = not state.active and desktop.is_game_focused(state.window)
        output.release_all()
        status = "VISUAL AI EVALUATING" if context.evaluation else "VISUAL AI TRAINING"
        if not active:
            status = "VISUAL AI PAUSED - focus Roblox, then F8"
        state = state.clear_tracking(
            active=active, retry_recovery_at=0.0, cooldown_until=0.0, status=status
        )
    return state


def capture_view(
    context: DrivingContext, state: DrivingState, capture
) -> tuple[DrivingState, dict | None, bool]:
    area = desktop.client_capture_area(state.window) if state.window else None
    if area is None:
        context.output.release_all()
        window = desktop.find_game_window(context.settings["window_title"]) or state.window
        context.output.window = window
        status = (
            "VISUAL AI WAITING - Roblox unavailable"
            if state.active
            else "VISUAL AI PAUSED - focus Roblox, then F8"
        )
        return state.clear_tracking(window=window, status=status), None, False
    focused = desktop.is_game_focused(state.window)
    if focused:
        try:
            return replace(state, frame=desktop.capture_frame(capture, area)), area, True
        except (OSError, mss.exception.ScreenShotError):
            context.output.release_all()
            return (
                state.clear_tracking(status="VISUAL AI WAITING - screen capture failed"),
                None,
                False,
            )
    if state.active:
        context.output.release_all()
        state = state.clear_tracking(status="VISUAL AI WAITING - focus Roblox to resume")
    return state, area, False


def recover_vehicle(context: DrivingContext, state: DrivingState, capture) -> bool:
    try:
        return recover_car(
            capture, state.window, context.keys, context.output, context.rng.randrange(4)
        )
    except (OSError, mss.exception.ScreenShotError):
        return False


def missing_motion(
    context: DrivingContext, state: DrivingState, capture, now: float, menu_open: bool
) -> DrivingState:
    context.output.release_all()
    lost_transition = state.lost_transition
    if state.last_state is not None and not menu_open:
        lost_transition = (state.last_state, state.last_action)
    state = replace(
        state,
        onroad_streak=0.0,
        last_state=None,
        stuck_since=None,
        lost_transition=lost_transition,
        missing_since=state.missing_since or now,
    )
    missing_duration = now - state.missing_since
    if missing_duration >= 0.6 and state.lost_transition is not None:
        lost_state, lost_action = state.lost_transition
        learner = state.learner
        if context.evaluation:
            context.evidence.record_step(
                lost_action, state.frame, None, None, -3.0, True, now - state.action_since
            )
        else:
            learner = learner.observe(
                (lost_state, lost_action, -3.0, lost_state, True), context.rng, context.policy_path
            )
        state = replace(state, learner=learner, lost_transition=None)
    if now < state.retry_recovery_at or missing_duration < 0.6:
        return replace(state, status="VISUAL AI WAITING - selecting a car")
    attempts = state.recovery_attempts + int(context.evaluation)
    if missing_duration >= 20.0:
        respawned = respawn_character(state.window, context.keys, context.output)
        missing_since = time.monotonic()
        status = (
            "VISUAL AI RECOVERING - character respawn requested"
            if respawned
            else "VISUAL AI WAITING - focus Roblox"
        )
        return replace(
            state,
            missing_since=missing_since,
            retry_recovery_at=missing_since + 5.0,
            recovery_attempts=attempts,
            status=status,
        )
    recovered = recover_vehicle(context, state, capture)
    status = (
        "VISUAL AI TRAINING - car selected"
        if recovered
        else "VISUAL AI RETRYING - car selection failed"
    )
    return replace(
        state,
        retry_recovery_at=time.monotonic() + (3.0 if recovered else 2.0),
        recovery_attempts=attempts,
        status=status,
    )


def complete_transition(
    context: DrivingContext,
    state: DrivingState,
    observation: np.ndarray,
    road: tuple | None,
    moving: bool,
    hud_visible: bool,
    hud_motion: bool | None,
    stuck: bool,
    now: float,
) -> DrivingState:
    alignment = (
        road_state(road, state.frame.shape[1], context.settings["car_x_fraction"])[1]
        if road is not None
        else None
    )
    driving_motion = moving
    if context.free_exploration:
        driving_motion = (
            hud_motion if hud_motion is not None else (state.scene_motion if hud_visible else None)
        )
    reward = transition_reward(observation, state.last_state, driving_motion, alignment, False)
    seen_views = state.seen_views
    if context.free_exploration and not hud_visible:
        small_view = cv2.resize(state.frame, (16, 9), interpolation=cv2.INTER_AREA)
        signature = hash((small_view // 32).tobytes())
        if signature not in seen_views:
            seen_views = seen_views | {signature}
            if len(seen_views) > 200000:
                seen_views = seen_views - {next(iter(seen_views))}
            reward += 0.30
    aligned_motion = driving_motion and alignment is not None and alignment <= 0.25
    streak = state.onroad_streak + now - state.action_since if aligned_motion else 0.0
    learner = replace(state.learner, best_streak=max(state.learner.best_streak, streak))
    reward = -2.0 if stuck else reward
    if context.evaluation:
        context.evidence.record_step(
            state.last_action,
            state.frame,
            moving,
            alignment,
            reward,
            stuck,
            now - state.action_since,
        )
    else:
        transition = (state.last_state, state.last_action, reward, observation, stuck)
        learner = learner.observe(transition, context.rng, context.policy_path)
    return replace(
        state,
        learner=learner,
        seen_views=seen_views,
        onroad_streak=streak,
        last_state=None,
        cooldown_until=now,
    )


def known_motion(
    context: DrivingContext,
    state: DrivingState,
    capture,
    road: tuple | None,
    moving: bool,
    hud_visible: bool,
    hud_motion: bool | None,
    now: float,
    area: dict,
) -> DrivingState:
    observation = visual_state(state.frame, moving, context.free_exploration)
    stuck_since = None if moving else (state.stuck_since or now)
    stuck = stuck_since is not None and now - stuck_since >= 20.0
    state = replace(state, missing_since=None, lost_transition=None, stuck_since=stuck_since)
    action_elapsed = now - state.action_since >= (0.25 if moving else 0.60)
    if state.last_state is not None and (action_elapsed or stuck):
        state = complete_transition(
            context, state, observation, road, moving, hud_visible, hud_motion, stuck, now
        )
    if stuck and now >= state.retry_recovery_at:
        context.output.release_all()
        recovered = recover_vehicle(context, state, capture)
        retry_at = time.monotonic()
        if recovered:
            return replace(
                state,
                stuck_since=None,
                retry_recovery_at=0.0,
                cooldown_until=retry_at + 0.5,
                status="VISUAL AI TRAINING - car respawned",
                recovery_attempts=state.recovery_attempts + int(context.evaluation),
            )
        return replace(
            state,
            stuck_since=retry_at,
            retry_recovery_at=retry_at + 2.0,
            status="VISUAL AI RETRYING - recovery failed",
            recovery_attempts=state.recovery_attempts + int(context.evaluation),
        )
    if state.last_state is not None or now < state.cooldown_until:
        return state
    action, selection = select_action(
        observation,
        state.learner,
        context.actions,
        context.rng,
        context.free_exploration,
        context.evaluation,
        moving,
        hud_visible,
        hud_motion,
    )
    if context.evaluation:
        # Preserve the attempted action even when the OS rejects input afterward.
        context.evidence.record_action(action, selection, hud_visible, hud_motion)
    names = execute_action(
        action, context.actions, context.codes, state.window, area, context.keys, context.output
    )
    label = (
        ("GAME EVALUATION" if context.evaluation else "GAME EXPLORER")
        if context.free_exploration
        else ("ROAD EVALUATION" if context.evaluation else "ROAD GOAL")
    )
    count = context.evidence.totals.steps if context.evaluation else state.learner.steps
    status = f"{label} - {count} steps | road {state.onroad_streak:.0f}s | best {state.learner.best_streak:.0f}s | {names}"
    return replace(
        state, last_state=observation, last_action=action, action_since=now, status=status
    )


def advance_driving(
    context: DrivingContext, state: DrivingState, capture, preview_road: tuple | None, area: dict
) -> DrivingState:
    now = time.monotonic()
    menu_open = car_tile(state.frame) is not None or car_category_icon(state.frame) is not None
    road = None if menu_open else preview_road
    if now - state.motion_at >= 0.4:
        state = replace(
            state,
            scene_motion=scene_moving(state.motion_frame, state.frame),
            motion_frame=state.frame.copy(),
            motion_at=now,
        )
    if road is not None:
        state = replace(state, last_road_at=now)
    hud_visible = not menu_open and speed_hud_visible(state.frame)
    hud_motion = speed_moving(state.frame) if hud_visible else None
    moving = hud_motion
    recently_saw_road = state.last_road_at is not None and now - state.last_road_at < 3.0
    if moving is None and (context.free_exploration or recently_saw_road):
        moving = state.scene_motion
    offroad = context.free_exploration and not menu_open and road is None and hud_motion is None
    state = replace(state, offroad_since=(state.offroad_since or now) if offroad else None)
    if offroad and now - state.offroad_since >= 30.0 and now >= state.retry_recovery_at:
        context.output.release_all()
        respawn_character(state.window, context.keys, context.output)
        state = replace(
            state,
            retry_recovery_at=time.monotonic() + 5.0,
            offroad_since=time.monotonic(),
            recovery_attempts=state.recovery_attempts + int(context.evaluation),
        )
        moving = None
    if moving is None:
        return missing_motion(context, state, capture, now, menu_open)
    return known_motion(context, state, capture, road, moving, hud_visible, hud_motion, now, area)


def ai_drive(free_exploration: bool = False, evaluation: bool = False) -> None:
    """Run paused by default; evaluation records evidence without updating the checkpoint."""
    settings, codes = load_settings()
    if settings["road_sample_lab"] is None:
        raise RuntimeError("Run python -m roblinforce calibrate before driving.")
    desktop.initialize_desktop()
    actions = FREE_ACTIONS if free_exploration else AI_ACTIONS
    policy_path = FREE_POLICY if free_exploration else VISUAL_POLICY
    if evaluation and not policy_path.exists():
        raise RuntimeError("Train this mode before evaluating its checkpoint.")
    learner = VisualLearner.load(policy_path, 9217 if free_exploration else 3601, len(actions))
    window = desktop.find_game_window(settings["window_title"])
    keys = desktop.PhysicalKeys(codes, stop_enabled=not free_exploration)
    output = desktop.KeyOutput(window)
    state = DrivingState(learner, window, np.zeros((720, 1280, 3), dtype=np.uint8))
    evidence = None
    try:
        keys.start()
        evidence = EvaluationLog(policy_path, actions) if evaluation else None
        context = DrivingContext(
            actions,
            policy_path,
            settings,
            codes,
            keys,
            output,
            random.Random(),
            evidence,
            free_exploration,
            evaluation,
        )
        with mss.MSS() as capture:
            cv2.namedWindow(PREVIEW_TITLE, cv2.WINDOW_AUTOSIZE)
            while True:
                state = poll_controls(context, state)
                state, area, focused = capture_view(context, state, capture)
                road = screen_road(state.frame, settings) if focused else None
                if state.active and focused:
                    state = advance_driving(context, state, capture, road, area)
                show_preview(state.frame, road, state.status, free_exploration)
                if preview_closed(free_exploration):
                    break
                if area is None:
                    time.sleep(0.1)
    finally:
        with ExitStack() as cleanup:
            if evidence is not None:
                cleanup.callback(evidence.close, state.recovery_attempts)
            elif not evaluation:
                cleanup.callback(state.learner.save, policy_path)
            cleanup.callback(cv2.destroyAllWindows)
            cleanup.callback(keys.close)
            cleanup.callback(output.release_all)
