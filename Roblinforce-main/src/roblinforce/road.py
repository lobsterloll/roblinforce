"""Calibrated road color and lane markings guide conservative steering.

The sampled road must continue through several horizontal bands. Lane markings
can provide a stronger estimate, but yellow lines must remain left of the car.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

RoadEstimate = Tuple[float, List[Tuple[int, float]]]


def _road_mask(frame: np.ndarray, settings: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, int]:
    height = frame.shape[0]
    top, bottom = int(height * 0.17), int(height * 0.47)
    road_region = frame[top:bottom]
    sample = np.asarray(settings["road_sample_lab"], dtype=np.int32)
    lab = cv2.cvtColor(road_region, cv2.COLOR_BGR2LAB).astype(np.int32)
    hsv = cv2.cvtColor(road_region, cv2.COLOR_BGR2HSV)
    color_distance = np.sum((lab - sample) ** 2, axis=2)
    matches_asphalt = (color_distance <= int(settings["color_tolerance"]) ** 2) & (
        hsv[:, :, 1] < 90
    )
    mask = np.uint8(matches_asphalt) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return mask, hsv, top


def _marking_positions(
    hsv: np.ndarray,
    frame_size: Tuple[int, int],
    top: int,
    lower: tuple,
    upper: tuple,
    minimum_support: int,
) -> List[np.ndarray]:
    height, width = frame_size
    line_top = int(height * 0.20)
    marking_mask = cv2.inRange(hsv[line_top - top :], lower, upper)
    marking_mask = cv2.morphologyEx(marking_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    lines = cv2.HoughLinesP(
        marking_mask, 1, np.pi / 180, 20, minLineLength=max(30, int(height * 0.05)), maxLineGap=18
    )
    if lines is None:
        return []
    positions = []
    for x1, y1, x2, y2 in lines[:, 0]:
        y1, y2 = y1 + line_top, y2 + line_top
        if abs(y2 - y1) < height * 0.075:
            continue
        near_x = x1 + (x2 - x1) * (height * 0.44 - y1) / (y2 - y1)
        far_x = x1 + (x2 - x1) * (height * 0.36 - y1) / (y2 - y1)
        if width * 0.1 < near_x < width * 0.9:
            positions.append((near_x, far_x))
    clusters: List[list] = []
    for position in sorted(positions):
        if clusters and position[0] - clusters[-1][-1][0] < width * 0.025:
            clusters[-1] = [*clusters[-1], position]
        else:
            clusters.append([position])
    return [np.median(cluster, axis=0) for cluster in clusters if len(cluster) >= minimum_support]


def _marked_lane(
    mask: np.ndarray, hsv: np.ndarray, frame_size: Tuple[int, int], top: int, anchor: float
) -> Optional[RoadEstimate]:
    height, width = frame_size
    yellow = _marking_positions(hsv, frame_size, top, (12, 85, 115), (38, 255, 255), 3)
    white = _marking_positions(hsv, frame_size, top, (0, 0, 200), (179, 75, 255), 2)
    yellow_on_left = any(position[0] < anchor - width * 0.02 for position in yellow)
    yellow_on_right = any(position[0] > anchor + width * 0.02 for position in yellow)
    left = [position for position in yellow + white if position[0] < anchor]
    right = [position for position in yellow + white if position[0] > anchor]
    if not left or not right or yellow_on_right or (yellow and not yellow_on_left):
        return None
    left_mark = max(left, key=lambda position: position[0])
    right_mark = min(right, key=lambda position: position[0])
    near_width, far_width = right_mark - left_mark
    near_lane, far_lane = (left_mark + right_mark) / 2
    plausible_width = (
        width * 0.08 <= near_width <= width * 0.42
        and width * 0.025 <= far_width <= near_width * 1.2
    )
    if not plausible_width:
        return None
    centers = [(int(height * 0.44), near_lane), (int(height * 0.36), far_lane)]
    for y, x in centers:
        if not 10 <= int(x) < width - 10:
            return None
        patch = mask[y - top - 5 : y - top + 6, int(x) - 10 : int(x) + 11]
        if np.mean(patch > 0) < 0.5:
            return None
    return 0.65 * near_lane + 0.35 * far_lane, centers


def _road_runs(band: np.ndarray, width: int) -> List[Tuple[int, int]]:
    edges = np.diff(np.pad(band, (1, 1)))
    raw_runs = zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    runs: List[Tuple[int, int]] = []
    for left, right in raw_runs:
        if runs and left - runs[-1][1] <= 0.025 * width:
            runs[-1] = (runs[-1][0], right)
        else:
            runs.append((left, right))
    return runs


def _color_road(
    mask: np.ndarray, frame_size: Tuple[int, int], top: int, car_x: float
) -> Optional[RoadEstimate]:
    height, width = frame_size
    centers: List[Tuple[int, float]] = []
    anchor = car_x
    previous_width = None
    for fraction in (0.44, 0.38, 0.32, 0.26):
        y = int(height * fraction)
        band = (mask[y - top - 3 : y - top + 4].mean(axis=0) > 150).astype(np.int8)
        candidates = []
        for left, right in _road_runs(band, width):
            road_width, center = right - left, (left + right) / 2
            if not centers and not left <= car_x <= right:
                continue
            if 0.04 * width <= road_width <= 0.80 * width and abs(center - anchor) <= 0.24 * width:
                candidates.append((center, road_width))
        if not candidates:
            continue
        center, road_width = min(candidates, key=lambda candidate: abs(candidate[0] - anchor))
        # A sudden widening often means an intersection rather than a continuous road.
        if previous_width is not None and road_width > 1.7 * previous_width:
            continue
        centers.append((y, center))
        previous_width, anchor = road_width, center
    if len(centers) < 3 or centers[0][0] - centers[-1][0] < 0.10 * height:
        return None
    return 0.65 * centers[0][1] + 0.35 * centers[-1][1], centers


def road_center(frame: np.ndarray, settings: Dict[str, Any]) -> Optional[RoadEstimate]:
    """Return a target x coordinate and supporting (y, x) road centers."""
    if settings["road_sample_lab"] is None:
        return None
    frame_size = frame.shape[:2]
    mask, hsv, top = _road_mask(frame, settings)
    car_x = frame_size[1] * float(settings["car_x_fraction"])
    marked_lane = _marked_lane(mask, hsv, frame_size, top, car_x)
    if marked_lane is not None:
        return marked_lane
    return _color_road(mask, frame_size, top, car_x)


def draw_preview(
    frame: np.ndarray,
    settings: Dict[str, Any],
    result: Optional[RoadEstimate],
    status: str,
    controls: str = "F8 start/pause | F9 stop | physical driving key takes over | Q quit",
) -> Tuple[np.ndarray, float]:
    """Annotate a copy so overlays cannot contaminate the next vision estimate."""
    preview = frame.copy()
    height, width = preview.shape[:2]
    car_x = int(width * float(settings["car_x_fraction"]))
    cv2.line(preview, (car_x, int(height * 0.17)), (car_x, int(height * 0.48)), (0, 255, 255), 2)
    if result:
        target, centers = result
        for y, x in centers:
            cv2.circle(preview, (int(x), y), 7, (0, 255, 0), -1)
        cv2.line(
            preview, (car_x, int(height * 0.44)), (int(target), centers[-1][0]), (0, 255, 0), 2
        )
    cv2.rectangle(preview, (0, 0), (width, 72), (0, 0, 0), -1)
    cv2.putText(preview, status, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        preview,
        controls,
        (15, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    scale = min(1.0, 1000 / width)
    return cv2.resize(preview, None, fx=scale, fy=scale), scale
