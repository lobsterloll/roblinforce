"""Record supervised examples, train nearest neighbors, or drive with steering."""

from datetime import datetime
from pathlib import Path
from contextlib import ExitStack
import random
import time

import cv2
import mss
import numpy as np

from . import desktop
from .settings import load_settings
from .road import draw_preview, road_center
from .config import DATA, MODEL, STEERING_POLICY
from .learning import predict, road_state, save_policy, update_policy
from .perception import features, speed_moving


def save_samples(vectors: list, labels: list) -> Path | None:
    if len(labels) == 0:
        return None
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"drive-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.npz"
    np.savez_compressed(
        path, x=np.asarray(vectors, dtype=np.float32), y=np.asarray(labels, dtype=np.int8)
    )
    return path


def record() -> None:
    desktop.initialize_desktop()
    settings, codes = load_settings()
    hwnd = desktop.find_game_window(settings["window_title"])
    if not hwnd:
        raise RuntimeError("Open Roblox before recording.")
    keys = desktop.PhysicalKeys(codes)
    keys.start()
    vectors, labels = [], []
    saved = 0
    try:
        with mss.MSS() as capture:
            while not keys.stop.is_set():
                keys.check_health()
                area = desktop.client_capture_area(hwnd)
                if area and desktop.is_game_focused(hwnd):
                    left = desktop.physical_down(codes["left"])
                    right = desktop.physical_down(codes["right"])
                    moving = desktop.physical_down(codes["accelerate"])
                    if moving and not (left and right):
                        frame = desktop.capture_frame(capture, area)
                        vectors.append(features(frame))
                        labels.append(int(right) - int(left))
                        if len(labels) >= 1000:
                            save_samples(vectors, labels)
                            saved += len(labels)
                            vectors, labels = [], []
                time.sleep(0.20)
    finally:
        keys.close()
    path = save_samples(vectors, labels)
    saved += len(labels)
    print(f"Saved {saved} driving examples" + (f"; last file: {path}" if path else "."))


def train() -> None:
    files = sorted(DATA.glob("drive-*.npz")) if DATA.exists() else []
    if not files:
        raise RuntimeError("No recordings yet. Run: python -m roblinforce record")
    rng = random.Random(7)
    buckets = {-1: [], 0: [], 1: []}
    counts = {-1: 0, 0: 0, 1: 0}
    for path in files:
        with np.load(path, allow_pickle=False) as session:
            for vector, label in zip(session["x"], session["y"]):
                label = int(label)
                counts[label] += 1
                bucket = buckets[label]
                # Reservoir sampling caps each class without favoring earlier sessions.
                if len(bucket) < 1500:
                    bucket.append(vector)
                else:
                    slot = rng.randrange(counts[label])
                    if slot < 1500:
                        bucket[slot] = vector
    if min(counts.values()) < 20:
        raise RuntimeError(
            f"Record at least 20 examples of left, straight, and right driving. Current counts: {counts}"
        )
    model_x = np.asarray(
        [vector for label in (-1, 0, 1) for vector in buckets[label]], dtype=np.float32
    )
    model_y = np.asarray([label for label in (-1, 0, 1) for _ in buckets[label]], dtype=np.int8)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODEL, x=model_x, y=model_y)
    print(
        f"Imitation model saved: {len(model_y)} examples. Left/straight/right: {counts[-1]}/{counts[0]}/{counts[1]}"
    )


def drive(self_training: bool = False) -> None:
    settings, codes = load_settings()
    if settings["road_sample_lab"] is None:
        raise RuntimeError("Run python -m roblinforce calibrate before driving.")
    desktop.initialize_desktop()
    if self_training:
        q = np.zeros((9, 5, 3), dtype=np.float32)
        visits = np.zeros_like(q, dtype=np.int32)
        if STEERING_POLICY.exists():
            with np.load(STEERING_POLICY, allow_pickle=False) as saved:
                q = saved["q"].astype(np.float32)
                visits = saved["visits"].astype(np.int32)
            if q.shape != (9, 5, 3) or visits.shape != q.shape:
                raise RuntimeError("The saved steering policy has an invalid shape.")
        rng = random.Random()
        last_state = None
        last_action = 0
        last_score = 0.0
        stuck_since = None
    else:
        if not MODEL.exists():
            raise RuntimeError("No model yet. Record your driving, then run train-imitation.")
        with np.load(MODEL, allow_pickle=False) as model:
            model_x = model["x"].astype(np.float32)
            model_y = model["y"].astype(np.int8)
    hwnd = desktop.find_game_window(settings["window_title"])
    if not hwnd:
        raise RuntimeError("Open Roblox before driving.")
    keys = desktop.PhysicalKeys(codes)
    output = desktop.KeyOutput(hwnd)
    keys.start()
    active = False
    status = "PAUSED - focus Roblox, then F8"
    last_prediction = 0.0
    direction = 0
    frame = None
    try:
        with mss.MSS() as capture:
            area = desktop.client_capture_area(hwnd)
            if area is None:
                raise RuntimeError("Roblox is minimized or its game view is too small.")
            frame = desktop.capture_frame(capture, area)
            cv2.namedWindow("Driving Empire preview", cv2.WINDOW_AUTOSIZE)
            while True:
                keys.check_health()
                if keys.stop.is_set():
                    keys.stop.clear()
                    keys.toggle.clear()
                    active = False
                    output.release_all()
                    status = "STOPPED - all keys released"
                if keys.takeover.is_set():
                    keys.takeover.clear()
                    keys.toggle.clear()
                    if self_training and active and last_state is not None:
                        q, visits = update_policy(q, visits, last_state, last_action, -0.5)
                    active = False
                    output.release_all()
                    status = "MANUAL TAKEOVER - F8 to resume"
                if keys.toggle.is_set():
                    keys.toggle.clear()
                    active = not active and desktop.is_game_focused(hwnd)
                    if self_training:
                        last_state = None
                        stuck_since = None
                        last_prediction = 0.0
                    if not active:
                        output.release_all()
                    status = (
                        "SELF-TRAINING"
                        if active and self_training
                        else ("ACTIVE" if active else "PAUSED - focus Roblox, then F8")
                    )
                area = desktop.client_capture_area(hwnd)
                if area is None:
                    active = False
                    output.release_all()
                    status = "PAUSED - Roblox unavailable"
                    time.sleep(0.1)
                    continue
                focused = desktop.is_game_focused(hwnd)
                if focused:
                    frame = desktop.capture_frame(capture, area)
                road = road_center(frame, settings)
                if active:
                    if not focused or road is None:
                        if self_training and focused and road is None and last_state is not None:
                            q, visits = update_policy(q, visits, last_state, last_action, -1.0)
                            last_state = None
                        active = False
                        output.release_all()
                        status = "PAUSED - focus or road lost"
                    else:
                        now = time.monotonic()
                        if self_training:
                            state, score, baseline, error = road_state(
                                road, frame.shape[1], float(settings["car_x_fraction"])
                            )
                            moving = speed_moving(frame)
                            if moving is None:
                                active = False
                                output.release_all()
                                status = "PAUSED - speed display missing"
                            elif abs(error) > 0.75:
                                if last_state is not None:
                                    q, visits = update_policy(
                                        q, visits, last_state, last_action, -0.5
                                    )
                                    last_state = None
                                active = False
                                output.release_all()
                                status = "PAUSED - too far from road path"
                            else:
                                stuck_since = None if moving else (stuck_since or now)
                                if stuck_since is not None and now - stuck_since >= 5.0:
                                    if last_state is not None:
                                        q, visits = update_policy(
                                            q, visits, last_state, last_action, -0.5
                                        )
                                        last_state = None
                                    output.release_all()
                                    active = False
                                    status = "PAUSED - vehicle may be stuck"
                            if active and now - last_prediction >= 0.40:
                                if last_state is not None and moving:
                                    reward = 2.0 * (last_score - score) - 0.005 * abs(last_action)
                                    q, visits = update_policy(
                                        q, visits, last_state, last_action, reward, state
                                    )
                                    if int(visits.sum()) % 50 == 0:
                                        save_policy(q, visits)
                                direction = baseline
                                if abs(error) < 0.35 and int(visits[state].sum()) >= 10:
                                    direction = int(np.argmax(q[state])) - 1
                                if abs(error) < 0.15 and rng.random() < 0.08:
                                    direction = rng.choice((-1, 0, 1))
                                last_state, last_action, last_score = state, direction, score
                                last_prediction = now
                        elif now - last_prediction >= 0.15:
                            direction, similarity, agreement = predict(
                                features(frame), model_x, model_y
                            )
                            last_prediction = now
                            if similarity < 0.85 or agreement < 5:
                                active = False
                                output.release_all()
                                status = "PAUSED - unfamiliar view"
                        if active:
                            wanted = set()
                            if now % 0.8 < float(settings["throttle_duty"]) * 0.8:
                                wanted.add(codes["accelerate"])
                            steer_now = (
                                now - last_prediction < 0.08 if self_training else now % 0.5 < 0.08
                            )
                            if direction and steer_now:
                                wanted.add(codes["left"] if direction < 0 else codes["right"])
                            output.set_pressed(wanted)
                            if self_training:
                                status = f"SELF-TRAINING - {int(visits.sum())} learned steps"
                            else:
                                status = f"ACTIVE - learned steering: {('LEFT', 'STRAIGHT', 'RIGHT')[direction + 1]}"
                preview, _ = draw_preview(frame, settings, road, status)
                cv2.imshow("Driving Empire preview", preview)
                key = cv2.waitKey(20) & 0xFF
                if (
                    key == ord("q")
                    or cv2.getWindowProperty("Driving Empire preview", cv2.WND_PROP_VISIBLE) < 1
                ):
                    break
    finally:
        with ExitStack() as cleanup:
            if self_training:
                cleanup.callback(save_policy, q, visits)
            cleanup.callback(cv2.destroyAllWindows)
            cleanup.callback(keys.close)
            cleanup.callback(output.release_all)
