"""Interruptible screen-only menu actions for recovering a vehicle."""

import time
from typing import Callable

import numpy as np

from . import desktop
from .perception import car_category_icon, car_tile, speed_moving


def can_control_game(hwnd, keys) -> bool:
    """Check physical interruption between each step of a menu sequence."""
    keys.check_health()
    interrupted = keys.stop.is_set() or keys.takeover.is_set() or keys.toggle.is_set()
    return desktop.is_game_focused(hwnd) and not interrupted


def click_game(hwnd, area: dict, point: tuple[int, int], keys, button: str = "left") -> bool:
    if not can_control_game(hwnd, keys):
        return False
    return desktop.click_game_point(hwnd, area, point, button)


def scroll_game(hwnd, area: dict, point: tuple[int, int], keys, amount: int) -> bool:
    if not can_control_game(hwnd, keys):
        return False
    return desktop.scroll_game_point(hwnd, area, point, amount)


def controlled_view(capture, hwnd, keys) -> tuple[dict, np.ndarray] | None:
    if not can_control_game(hwnd, keys):
        return None
    area = desktop.client_capture_area(hwnd)
    if area is None:
        return None
    return area, desktop.capture_frame(capture, area)


def wait_for_menu_view(
    capture, hwnd, keys, ready: Callable[[np.ndarray], bool]
) -> tuple[dict, np.ndarray] | None:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        view = controlled_view(capture, hwnd, keys)
        if view is None:
            return None
        if ready(view[1]):
            return view
        time.sleep(0.10)
    return None


def confirm_vehicle_spawn(capture, hwnd, keys, output) -> bool:
    clicked_at = time.monotonic()
    deadline = clicked_at + 10.0
    closed_menu = False
    hud_seen = 0
    while time.monotonic() < deadline:
        view = controlled_view(capture, hwnd, keys)
        if view is None:
            return False
        _, frame = view
        menu_open = car_tile(frame) is not None
        if not menu_open and (
            speed_moving(frame) is not None or time.monotonic() - clicked_at >= 1.0
        ):
            hud_seen += 1
            # Two observations avoid treating the menu's closing animation as a spawned car.
            if hud_seen >= 2:
                return True
        else:
            hud_seen = 0
        if menu_open and not closed_menu and time.monotonic() - clicked_at >= 0.7:
            output.set_pressed({desktop.virtual_key("T")})
            time.sleep(0.10)
            output.release_all()
            closed_menu = True
        time.sleep(0.10)
    return False


def recover_car(capture, hwnd, keys, output, choice: int = 0) -> bool:
    output.release_all()
    view = controlled_view(capture, hwnd, keys)
    if view is None:
        return False
    _, frame = view
    menu_visible = car_tile(frame) is not None or car_category_icon(frame) is not None
    menu_keys = () if menu_visible else (("F", "T") if speed_moving(frame) is not None else ("T",))
    for name in menu_keys:
        output.set_pressed({desktop.virtual_key(name)})
        time.sleep(0.10)
        output.release_all()

        def menu_ready(image: np.ndarray) -> bool:
            if name == "F":
                return speed_moving(image) is None
            return car_tile(image) is not None or car_category_icon(image) is not None

        if wait_for_menu_view(capture, hwnd, keys, menu_ready) is None:
            return False
    view = controlled_view(capture, hwnd, keys)
    if view is None:
        return False
    area, frame = view
    category = car_category_icon(frame) if car_tile(frame) is None else None
    if category is not None:
        if not click_game(hwnd, area, category, keys):
            return False
        view = wait_for_menu_view(capture, hwnd, keys, lambda image: car_tile(image) is not None)
        if view is None:
            return False
        area, frame = view
    tile = car_tile(frame, choice)
    if tile is None or not click_game(hwnd, area, tile, keys):
        return False
    return confirm_vehicle_spawn(capture, hwnd, keys, output)


def respawn_character(hwnd, keys, output) -> bool:
    output.release_all()
    for code in (0x1B, ord("R"), 0x0D):  # Roblox menu, Respawn, confirmation
        if not can_control_game(hwnd, keys):
            return False
        output.set_pressed({code})
        time.sleep(0.10)
        output.release_all()
        time.sleep(0.35)
    return True
