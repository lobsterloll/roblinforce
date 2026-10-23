"""Sample road color in a preview without creating an input controller."""

import json
from typing import Any, Dict, Tuple

import cv2
import mss
import numpy as np

from . import desktop
from .perception import screen_road
from .road import draw_preview
from .settings import SETTINGS_PATH, load_settings

PREVIEW_TITLE = "Driving Empire road calibration"
CALIBRATION_CONTROLS = "Click plain gray road to save calibration | Q or close window to quit"


def calibrate_road(patch: np.ndarray, settings: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Reject saturated scenery and return a new configuration for plain asphalt."""
    saturation = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 1]
    if np.median(saturation) >= 90:
        return settings, "NOT ROAD - click plain gray asphalt"
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    sample_lab = np.median(lab.reshape(-1, 3), axis=0).astype(int).tolist()
    return {**settings, "road_sample_lab": sample_lab}, "CALIBRATED - saved road color"


def run() -> None:
    """Display captured road samples; only preview mouse clicks change settings."""
    settings, _ = load_settings()
    desktop.initialize_desktop()
    window = desktop.find_game_window(settings["window_title"])
    if not window:
        raise RuntimeError("Open Roblox before calibrating the road.")
    pending_click = None
    preview_scale = 1.0
    status = "Click a clear patch of gray road"

    def click(event: int, x: int, y: int, _flags: int, _data: Any) -> None:
        nonlocal pending_click
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click = (x, y)

    try:
        with mss.MSS() as capture:
            area = desktop.client_capture_area(window)
            if area is None:
                raise RuntimeError("Roblox is minimized or its game view is too small.")
            frame = desktop.capture_frame(capture, area)
            cv2.namedWindow(PREVIEW_TITLE, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(PREVIEW_TITLE, click)
            while True:
                area = desktop.client_capture_area(window)
                if area is None:
                    raise RuntimeError("Roblox became unavailable. Restore it and calibrate again.")
                # Preserve the last game image while the user clicks the preview.
                if desktop.is_game_focused(window):
                    frame = desktop.capture_frame(capture, area)
                if pending_click is not None:
                    x = int(pending_click[0] / preview_scale)
                    y = int(pending_click[1] / preview_scale)
                    pending_click = None
                    height, width = frame.shape[:2]
                    if 8 <= x < width - 8 and 8 <= y < height - 8:
                        updated_settings, status = calibrate_road(
                            frame[y - 8 : y + 9, x - 8 : x + 9], settings
                        )
                        if updated_settings is not settings:
                            SETTINGS_PATH.write_text(
                                json.dumps(updated_settings, indent=2) + "\n", encoding="utf-8"
                            )
                            settings = updated_settings
                road = screen_road(frame, settings)
                preview, preview_scale = draw_preview(
                    frame, settings, road, status, CALIBRATION_CONTROLS
                )
                cv2.imshow(PREVIEW_TITLE, preview)
                key = cv2.waitKey(20) & 0xFF
                if (
                    key == ord("q")
                    or cv2.getWindowProperty(PREVIEW_TITLE, cv2.WND_PROP_VISIBLE) < 1
                ):
                    break
    finally:
        cv2.destroyAllWindows()
