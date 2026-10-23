"""Quartz desktop adapter using logical coordinates and tagged synthetic events.

Public CGWindow APIs expose the full window rectangle, including its title bar.
Capture and pointer input deliberately use that same rectangle. MSS Retina images
are resized by desktop.capture_frame before vision or pointer targeting.
"""

import os
import threading
import time
from typing import Any, Dict, List, Optional

SYNTHETIC_EVENT_TAG = 0x524F424C
# Logical Windows codes map to the fixed macOS hardware key positions used by games.
MAC_KEY_CODES = {
    65: 0,
    83: 1,
    68: 2,
    70: 3,
    72: 4,
    71: 5,
    90: 6,
    88: 7,
    67: 8,
    86: 9,
    66: 11,
    81: 12,
    87: 13,
    69: 14,
    82: 15,
    89: 16,
    84: 17,
    49: 18,
    50: 19,
    51: 20,
    52: 21,
    54: 22,
    53: 23,
    57: 25,
    55: 26,
    56: 28,
    48: 29,
    79: 31,
    85: 32,
    73: 34,
    80: 35,
    76: 37,
    74: 38,
    75: 40,
    78: 45,
    77: 46,
    0x20: 49,
    0x0D: 36,
    0x1B: 53,
    0x10: 56,
    0x25: 123,
    0x27: 124,
    0x28: 125,
    0x26: 126,
    0x70: 122,
    0x71: 120,
    0x72: 99,
    0x73: 118,
    0x74: 96,
    0x75: 97,
    0x76: 98,
    0x77: 100,
    0x78: 101,
    0x79: 109,
    0x7A: 103,
    0x7B: 111,
}


class MacDesktop:
    def __init__(self) -> None:
        try:
            import AppKit
            import Quartz
        except ImportError as exc:
            raise RuntimeError(
                "Install macOS dependencies from requirements.txt (PyObjC Quartz and Cocoa)."
            ) from exc
        self.quartz = Quartz
        self.appkit = AppKit

    def initialize(self) -> None:
        quartz = self.quartz
        permissions = (
            (quartz.CGPreflightScreenCaptureAccess(), "Screen Recording"),
            (quartz.CGPreflightPostEventAccess(), "Accessibility"),
            (quartz.CGPreflightListenEventAccess(), "Input Monitoring"),
        )
        missing = [name for granted, name in permissions if not granted]
        if missing:
            names = ", ".join(missing)
            raise RuntimeError(
                f"Grant {names} to your terminal or Python app in System Settings > Privacy & Security, then restart it."
            )

    def _windows(self) -> List[Dict[str, Any]]:
        quartz = self.quartz
        options = (
            quartz.kCGWindowListOptionOnScreenOnly | quartz.kCGWindowListExcludeDesktopElements
        )
        windows = quartz.CGWindowListCopyWindowInfo(options, quartz.kCGNullWindowID) or []
        return [window for window in windows if window.get(quartz.kCGWindowLayer, 0) == 0]

    def find_game_window(self, title_fragment: str) -> Optional[int]:
        quartz = self.quartz
        for window in self._windows():
            title = window.get(quartz.kCGWindowName, "")
            owner = window.get(quartz.kCGWindowOwnerName, "")
            if title_fragment.casefold() in f"{owner} {title}".casefold():
                return int(window[quartz.kCGWindowNumber])
        return None

    def client_capture_area(self, window: int) -> Optional[Dict[str, int]]:
        quartz = self.quartz
        selected = next(
            (item for item in self._windows() if item[quartz.kCGWindowNumber] == window), None
        )
        if selected is None:
            return None
        bounds = selected[quartz.kCGWindowBounds]
        width, height = int(bounds["Width"]), int(bounds["Height"])
        if width < 400 or height < 300:
            return None
        return {"left": int(bounds["X"]), "top": int(bounds["Y"]), "width": width, "height": height}

    def is_game_focused(self, window: int) -> bool:
        quartz = self.quartz
        windows = self._windows()
        if (
            not windows
            or windows[0][quartz.kCGWindowNumber] != window
            or self.client_capture_area(window) is None
        ):
            return False
        application = self.appkit.NSWorkspace.sharedWorkspace().frontmostApplication()
        return (
            application is not None
            and application.processIdentifier() == windows[0][quartz.kCGWindowOwnerPID]
        )

    def physical_down(self, code: int) -> bool:
        quartz = self.quartz
        native_codes = (56, 60) if code == 0x10 else (MAC_KEY_CODES[code],)
        return any(
            quartz.CGEventSourceKeyState(quartz.kCGEventSourceStateHIDSystemState, native)
            for native in native_codes
        )

    def _post_event(self, event: Any) -> None:
        quartz = self.quartz
        if event is None:
            raise OSError("Quartz could not create an input event.")
        quartz.CGEventSetIntegerValueField(
            event, quartz.kCGEventSourceUserData, SYNTHETIC_EVENT_TAG
        )
        quartz.CGEventPost(quartz.kCGHIDEventTap, event)

    def send_key(self, code: int, down: bool) -> None:
        self._post_event(self.quartz.CGEventCreateKeyboardEvent(None, MAC_KEY_CODES[code], down))

    def click(self, x: int, y: int, button: str = "left") -> None:
        quartz = self.quartz
        if button == "left":
            down, up, native_button = (
                quartz.kCGEventLeftMouseDown,
                quartz.kCGEventLeftMouseUp,
                quartz.kCGMouseButtonLeft,
            )
        else:
            down, up, native_button = (
                quartz.kCGEventRightMouseDown,
                quartz.kCGEventRightMouseUp,
                quartz.kCGMouseButtonRight,
            )
        self._post_event(quartz.CGEventCreateMouseEvent(None, down, (x, y), native_button))
        try:
            # Match the Windows dwell so game menus can observe a held mouse button.
            time.sleep(0.08)
        finally:
            self._post_event(quartz.CGEventCreateMouseEvent(None, up, (x, y), native_button))

    def scroll(self, x: int, y: int, amount: int) -> None:
        quartz = self.quartz
        self._post_event(
            quartz.CGEventCreateMouseEvent(
                None, quartz.kCGEventMouseMoved, (x, y), quartz.kCGMouseButtonLeft
            )
        )
        # Windows wheel deltas are multiples of 120; Quartz line units are not.
        lines = int(amount / 120) if abs(amount) >= 120 else int(amount)
        self._post_event(
            quartz.CGEventCreateScrollWheelEvent(None, quartz.kCGScrollEventUnitLine, 1, lines)
        )

    def keyboard_listener(self, owner: Any) -> "MacKeyboardListener":
        return MacKeyboardListener(self.quartz, owner)


class MacKeyboardListener:
    def __init__(self, quartz: Any, owner: Any) -> None:
        self.quartz = quartz
        self.owner = owner
        self.closed = threading.Event()
        self.run_loop = None
        self.tap = None
        self.callback = None

    def run(self) -> None:
        quartz = self.quartz
        logical_codes = {native: logical for logical, native in MAC_KEY_CODES.items()}
        logical_codes[60] = 0x10  # Either physical Shift key represents logical SHIFT.
        key_events = (quartz.kCGEventKeyDown, quartz.kCGEventKeyUp, quartz.kCGEventFlagsChanged)

        def callback(_proxy, event_type, event, _context):
            if event_type in (
                quartz.kCGEventTapDisabledByTimeout,
                quartz.kCGEventTapDisabledByUserInput,
            ):
                self.owner.error = RuntimeError(
                    "macOS disabled the keyboard safety listener. Restart the assistant."
                )
                self.owner.stop.set()
                self.closed.set()
                return event
            synthetic = (
                quartz.CGEventGetIntegerValueField(event, quartz.kCGEventSourceUserData)
                == SYNTHETIC_EVENT_TAG
            )
            synthetic = (
                synthetic
                or quartz.CGEventGetIntegerValueField(event, quartz.kCGEventSourceUnixProcessID)
                == os.getpid()
            )
            native = quartz.CGEventGetIntegerValueField(event, quartz.kCGKeyboardEventKeycode)
            code = logical_codes.get(native)
            if code is not None:
                down = event_type == quartz.kCGEventKeyDown
                if event_type == quartz.kCGEventFlagsChanged:
                    down = bool(
                        quartz.CGEventSourceKeyState(
                            quartz.kCGEventSourceStateHIDSystemState, native
                        )
                    )
                self.owner.handle_key(code, down, synthetic=synthetic)
            return event

        self.callback = (
            callback  # Keep the Python callback alive while Core Foundation owns the tap.
        )
        mask = sum(1 << event_type for event_type in key_events)
        self.tap = quartz.CGEventTapCreate(
            quartz.kCGSessionEventTap,
            quartz.kCGHeadInsertEventTap,
            quartz.kCGEventTapOptionListenOnly,
            mask,
            callback,
            None,
        )
        if self.tap is None:
            raise RuntimeError(
                "Keyboard safety listener needs macOS Input Monitoring and Accessibility permissions."
            )
        source = quartz.CFMachPortCreateRunLoopSource(None, self.tap, 0)
        self.run_loop = quartz.CFRunLoopGetCurrent()
        quartz.CFRunLoopAddSource(self.run_loop, source, quartz.kCFRunLoopCommonModes)
        quartz.CGEventTapEnable(self.tap, True)
        self.owner.ready.set()
        try:
            while not self.closed.is_set():
                quartz.CFRunLoopRunInMode(quartz.kCFRunLoopDefaultMode, 0.1, False)
        finally:
            quartz.CGEventTapEnable(self.tap, False)
            quartz.CFRunLoopRemoveSource(self.run_loop, source, quartz.kCFRunLoopCommonModes)
            quartz.CFMachPortInvalidate(self.tap)

    def close(self) -> None:
        self.closed.set()
        if self.run_loop is not None:
            self.quartz.CFRunLoopStop(self.run_loop)
