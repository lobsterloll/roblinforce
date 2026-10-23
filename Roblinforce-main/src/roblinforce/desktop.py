"""Shared desktop operations; native APIs are loaded only when driving starts.

Key codes use Windows virtual-key values as a platform-neutral logical format.
Capture rectangles and pointer coordinates are always desktop logical pixels.
"""

import sys
import threading
from typing import Any, Dict, Optional, Set, Tuple

_backend = None


def _create_backend() -> Any:
    if sys.platform == "win32":
        from .desktop_windows import WindowsDesktop

        return WindowsDesktop()
    if sys.platform == "darwin":
        from .desktop_macos import MacDesktop

        return MacDesktop()
    raise RuntimeError("Desktop driving supports Windows and macOS only.")


def _get_backend() -> Any:
    global _backend
    if _backend is None:
        _backend = _create_backend()
    return _backend


def initialize_desktop() -> None:
    """Check native permissions before starting capture or keyboard output."""
    _get_backend().initialize()


def virtual_key(name: str) -> int:
    name = name.upper()
    if len(name) == 1 and name.isascii() and name.isalnum():
        return ord(name)
    named = {
        "SPACE": 0x20,
        "SHIFT": 0x10,
        "ESC": 0x1B,
        "UP": 0x26,
        "DOWN": 0x28,
        "LEFT": 0x25,
        "RIGHT": 0x27,
    }
    if name in named:
        return named[name]
    if name.startswith("F") and name[1:].isdigit() and 1 <= int(name[1:]) <= 12:
        return 0x70 + int(name[1:]) - 1
    raise ValueError(f"Unsupported key: {name}")


def find_game_window(title_fragment: str) -> Optional[int]:
    return _get_backend().find_game_window(title_fragment)


def client_capture_area(window: int) -> Optional[Dict[str, int]]:
    return _get_backend().client_capture_area(window)


def is_game_focused(window: Optional[int]) -> bool:
    return bool(window is not None and _get_backend().is_game_focused(window))


def physical_down(code: int) -> bool:
    return _get_backend().physical_down(code)


def capture_frame(capture: Any, area: Dict[str, int]) -> Any:
    """Return BGR pixels at logical size, including on Retina displays."""
    import cv2
    import numpy as np

    frame = np.asarray(capture.grab(area))[:, :, :3].copy()
    expected_size = (area["width"], area["height"])
    if (frame.shape[1], frame.shape[0]) != expected_size:
        frame = cv2.resize(frame, expected_size, interpolation=cv2.INTER_AREA)
    return frame


def _pointer_position(
    window: int, area: Dict[str, int], point: Tuple[int, int]
) -> Optional[Tuple[int, int]]:
    if not is_game_focused(window) or client_capture_area(window) != area:
        return None
    x, y = point
    if not (0 <= x < area["width"] and 0 <= y < area["height"]):
        return None
    return area["left"] + int(x), area["top"] + int(y)


def click_game_point(
    window: int, area: Dict[str, int], point: Tuple[int, int], button: str = "left"
) -> bool:
    position = _pointer_position(window, area, point)
    if position is None:
        return False
    if button not in ("left", "right"):
        raise ValueError("Mouse button must be left or right.")
    _get_backend().click(*position, button=button)
    return True


def scroll_game_point(
    window: int, area: Dict[str, int], point: Tuple[int, int], amount: int
) -> bool:
    position = _pointer_position(window, area, point)
    if position is None:
        return False
    _get_backend().scroll(*position, amount)
    return True


class KeyOutput:
    """Track assistant-owned keys and refuse presses outside the game window."""

    def __init__(self, window: Optional[int] = None):
        self.window = window
        self.pressed: Set[int] = set()

    def send(self, vk: int, down: bool) -> None:
        if down and not is_game_focused(self.window):
            return
        _get_backend().send_key(vk, down)
        self.pressed = self.pressed | {vk} if down else self.pressed - {vk}

    def set_pressed(self, wanted: Set[int]) -> None:
        allowed = set(wanted) if is_game_focused(self.window) else set()
        first_error = None
        for key in sorted(self.pressed - allowed):
            try:
                self.send(key, False)
            except OSError as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
        for key in sorted(allowed - self.pressed):
            self.send(key, True)

    def release_all(self) -> None:
        self.set_pressed(set())


class PhysicalKeys:
    """Observe physical keys without treating assistant-generated events as takeover."""

    def __init__(self, codes: Dict[str, int], stop_enabled: bool = True):
        self.codes = dict(codes)
        self.stop_enabled = stop_enabled
        self.toggle = threading.Event()
        self.stop = threading.Event()
        self.takeover = threading.Event()
        self.ready = threading.Event()
        self.error: Optional[BaseException] = None
        self.thread: Optional[threading.Thread] = None
        self._down: Set[int] = set()
        self._listener: Any = None
        self._closing = threading.Event()
        self._driving_keys = {
            codes[name] for name in ("accelerate", "brake", "handbrake", "left", "right")
        } | {0x1B}

    def handle_key(self, code: int, down: bool, synthetic: bool = False) -> None:
        if synthetic:
            return
        if not down:
            self._down = self._down - {code}
            return
        if code in self._down:
            return
        self._down = self._down | {code}
        if self.stop_enabled and code == self.codes["stop"]:
            self.stop.set()
        elif code == self.codes["toggle"]:
            self.toggle.set()
        elif code in self._driving_keys:
            self.takeover.set()

    def start(self) -> None:
        self._listener = _get_backend().keyboard_listener(self)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        if not self.ready.wait(3):
            self.close()
            raise RuntimeError("Keyboard safety listener did not start.")
        self.check_health()

    def _run(self) -> None:
        try:
            self._listener.run()
            if not self._closing.is_set() and self.error is None:
                raise RuntimeError("Keyboard safety listener stopped unexpectedly.")
        except Exception as exc:
            self.error = exc
            self.stop.set()
        finally:
            self.ready.set()

    def check_health(self) -> None:
        if self.error is not None:
            raise RuntimeError(f"Keyboard safety listener failed: {self.error}") from self.error

    def close(self) -> None:
        self._closing.set()
        if self._listener is not None:
            self._listener.close()
        if self.thread is not None:
            self.thread.join(timeout=1)
