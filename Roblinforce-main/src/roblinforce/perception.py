"""Screen-only feature extraction for the experimental drivers."""

import cv2
import numpy as np

from .config import ASSETS

SPEED_ZERO = cv2.imread(str(ASSETS / "speed-zero.png"), cv2.IMREAD_GRAYSCALE)
HOG = cv2.HOGDescriptor((64, 32), (16, 16), (8, 8), (8, 8), 9)


def features(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    view = frame[int(height * 0.17) : int(height * 0.60), int(width * 0.10) : int(width * 0.90)]
    gray = cv2.cvtColor(cv2.resize(view, (64, 32)), cv2.COLOR_BGR2GRAY)
    # HOG keeps recorded steering examples independent of the window resolution.
    vector = HOG.compute(gray).reshape(-1).astype(np.float32)
    return vector / (np.linalg.norm(vector) + 1e-6)


def screen_road(frame: np.ndarray, settings: dict) -> tuple | None:
    if settings["road_sample_lab"] is None:
        return None
    height, width = frame.shape[:2]
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.int16)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sample = settings["road_sample_lab"]
    chroma = np.abs(lab[:, :, 1] - sample[1]) + np.abs(lab[:, :, 2] - sample[2])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = cv2.blur(cv2.magnitude(gx, gy), (15, 15))
    surface = (
        np.uint8((chroma < 23) & (hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 45) & (texture < 65)) * 255
    )
    surface = cv2.morphologyEx(surface, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    anchor = width * float(settings["car_x_fraction"])
    centers = []
    for fraction in (0.60, 0.53, 0.46, 0.39):
        y = int(height * fraction)
        band = (surface[y - 3 : y + 4].mean(axis=0) > 130).astype(np.int8)
        edges = np.diff(np.pad(band, (1, 1)))
        raw_runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
        runs = []
        for left, right in raw_runs:
            gap = left - runs[-1][1] if runs else None
            car_occlusion = bool(runs) and runs[-1][1] < anchor < left and gap <= 0.20 * width
            if runs and (gap <= 0.09 * width or car_occlusion):
                runs[-1] = (runs[-1][0], right)
            else:
                runs.append((left, right))
        candidates = []
        for left, right in runs:
            road_width = right - left
            center = (left + right) / 2
            if not centers and not left - 0.04 * width <= anchor <= right + 0.04 * width:
                continue
            if centers and abs(center - anchor) > 0.08 * width:
                continue
            if 0.12 * width <= road_width <= 0.95 * width and abs(center - anchor) < 0.30 * width:
                candidates.append((center, road_width))
        if not candidates:
            continue
        center, _ = min(candidates, key=lambda item: abs(item[0] - anchor))
        centers.append((y, center))
        anchor = center
    if len(centers) < 2:
        return None
    return 0.65 * centers[0][1] + 0.35 * centers[-1][1], centers


def scene_moving(previous: np.ndarray | None, frame: np.ndarray) -> bool:
    if previous is None or previous.shape != frame.shape:
        return False
    height, width = frame.shape[:2]
    region = (
        slice(int(height * 0.25), int(height * 0.60)),
        slice(int(width * 0.20), int(width * 0.80)),
    )
    before = cv2.cvtColor(cv2.resize(previous[region], (96, 48)), cv2.COLOR_BGR2GRAY)
    after = cv2.cvtColor(cv2.resize(frame[region], (96, 48)), cv2.COLOR_BGR2GRAY)
    return float(np.mean(np.abs(after.astype(np.int16) - before.astype(np.int16)))) > 5.0


def speed_hud_visible(frame: np.ndarray) -> bool:
    height, width = frame.shape[:2]
    hud = cv2.cvtColor(frame[int(height * 0.94) :, int(width * 0.72) :], cv2.COLOR_BGR2HSV)
    blue_bar = cv2.inRange(hud, (80, 100, 100), (110, 255, 255))
    return cv2.countNonZero(blue_bar) >= 50


def speed_moving(frame: np.ndarray) -> bool | None:
    height, width = frame.shape[:2]
    # Unknown is distinct from stopped: menus and HUD changes must not imply zero speed.
    if not speed_hud_visible(frame):
        return None
    left, right = int(width * 0.78), int(width * 0.94)
    top, bottom = int(height * 0.76), int(height * 0.96)
    hsv = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    white = np.uint8((hsv[:, :, 1] < 80) & (hsv[:, :, 2] > 190)) * 255
    count, _, components, _ = cv2.connectedComponentsWithStats(white)
    digits = []
    for x, y, digit_width, digit_height, area in components[1:count]:
        if (
            0.80 * width < x + left < 0.92 * width
            and 0.82 * height < y + top < 0.93 * height
            and 0.035 * height < digit_height < 0.07 * height
            and 0.008 * width < digit_width < 0.04 * width
            and area > 100 * (height / 1078) ** 2
        ):
            digits.append((x, y, digit_width, digit_height, area))
    if digits:
        reference_y = max(digits, key=lambda digit: digit[4])[1]
        digits = [digit for digit in digits if abs(digit[1] - reference_y) < 0.015 * height]
    if not 1 <= len(digits) <= 3:
        return None
    if len(digits) > 1:
        return True
    x, y, digit_width, digit_height, _ = digits[0]
    glyph = cv2.resize(
        white[y : y + digit_height, x : x + digit_width], (24, 32), interpolation=cv2.INTER_AREA
    )
    distance = np.abs(glyph.astype(np.int16) - SPEED_ZERO.astype(np.int16)).mean() / 255
    return bool(distance > 0.20)


def car_tile(frame: np.ndarray, choice: int = 0) -> tuple[int, int] | None:
    height, width = frame.shape[:2]
    left = int(width * 0.65)
    hsv = cv2.cvtColor(frame[: int(height * 0.45), left:], cv2.COLOR_BGR2HSV)
    purple = cv2.inRange(hsv, (125, 110, 85), (150, 255, 240))
    purple = cv2.morphologyEx(purple, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(purple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, bar_width, bar_height = cv2.boundingRect(contour)
        x += left
        if not (
            x > width * 0.60
            and x + bar_width > width * 0.90
            and width * 0.08 < bar_width < width * 0.32
            and height * 0.025 < bar_height < height * 0.08
            and bar_width > 5 * bar_height
        ):
            continue
        cards = [
            (int(x + col * bar_width), int(y + row * bar_width))
            for row in (1.60, 2.16)
            for col in (0.22, 0.78)
            if y + row * bar_width < height * 0.88
        ]
        if cards:
            return cards[choice % len(cards)]
    return None


def car_category_icon(frame: np.ndarray) -> tuple[int, int] | None:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame[: int(height * 0.75), int(width * 0.75) :], cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (90, 120, 100), (125, 255, 255))
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, icon_width, icon_height = cv2.boundingRect(contour)
        x += int(width * 0.75)
        if (
            x + icon_width > width * 0.95
            and width * 0.03 < icon_width < width * 0.12
            and height * 0.05 < icon_height < height * 0.15
        ):
            return (x + icon_width // 2, int(y + 1.62 * icon_height))
    return None


def visual_state(frame: np.ndarray, moving: bool, full_view: bool = False) -> np.ndarray:
    height, width = frame.shape[:2]
    view = (
        frame
        if full_view
        else frame[int(height * 0.12) : int(height * 0.83), int(width * 0.08) : int(width * 0.92)]
    )
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    size = (128, 72) if full_view else (80, 45)
    # Preserve pixel ordering and dimensions so existing checkpoints remain loadable.
    pixels = cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - 0.5
    return np.append(pixels.ravel(), np.float32(moving))
