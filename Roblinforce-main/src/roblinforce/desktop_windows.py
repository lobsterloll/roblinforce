"""Win32 window discovery, scan-code input, and a physical keyboard hook."""

import ctypes
from ctypes import wintypes
import time
from typing import Any, Dict, Optional

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

ULONG_PTR = ctypes.c_size_t
LRESULT = ctypes.c_ssize_t
HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
ENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class INPUTUNION(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", INPUTUNION)]


user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, ctypes.c_void_p, wintypes.DWORD]
user32.SetWindowsHookExW.restype = ctypes.c_void_p
user32.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
user32.CallNextHookEx.restype = LRESULT
user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
user32.GetMessageW.argtypes = [
    ctypes.POINTER(wintypes.MSG),
    ctypes.c_void_p,
    wintypes.UINT,
    wintypes.UINT,
]
user32.GetMessageW.restype = ctypes.c_int
user32.PostThreadMessageW.argtypes = [
    wintypes.DWORD,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
]
user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wintypes.UINT
user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
user32.MapVirtualKeyW.restype = wintypes.UINT
user32.EnumWindows.argtypes = [ENUMPROC, wintypes.LPARAM]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsIconic.argtypes = [wintypes.HWND]
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
user32.IsWindow.argtypes = [wintypes.HWND]
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.mouse_event.argtypes = [
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ULONG_PTR,
]
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetForegroundWindow.restype = wintypes.HWND
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = ctypes.c_void_p
kernel32.GetCurrentThreadId.restype = wintypes.DWORD


class WindowsDesktop:
    def initialize(self) -> None:
        user32.SetProcessDPIAware()

    def find_game_window(self, title_fragment: str) -> Optional[int]:
        matches = []

        @ENUMPROC
        def visit(window, _):
            if not user32.IsWindowVisible(window):
                return True
            title = ctypes.create_unicode_buffer(user32.GetWindowTextLengthW(window) + 1)
            user32.GetWindowTextW(window, title, len(title))
            if title_fragment.casefold() in title.value.casefold():
                matches.append(window)
            return True

        user32.EnumWindows(visit, 0)
        return matches[0] if matches else None

    def client_capture_area(self, window: int) -> Optional[Dict[str, int]]:
        if not window or not user32.IsWindow(window) or user32.IsIconic(window):
            return None
        bounds = wintypes.RECT()
        origin = wintypes.POINT(0, 0)
        if not user32.GetClientRect(window, ctypes.byref(bounds)):
            return None
        if not user32.ClientToScreen(window, ctypes.byref(origin)):
            return None
        width = bounds.right - bounds.left
        height = bounds.bottom - bounds.top
        if width < 400 or height < 300:
            return None
        return {"left": origin.x, "top": origin.y, "width": width, "height": height}

    def is_game_focused(self, window: int) -> bool:
        return (
            self.client_capture_area(window) is not None and user32.GetForegroundWindow() == window
        )

    def physical_down(self, code: int) -> bool:
        return bool(user32.GetAsyncKeyState(code) & 0x8000)

    def send_key(self, code: int, down: bool) -> None:
        scan = user32.MapVirtualKeyW(code, 0)
        flags = 0x0008 | (0 if down else 0x0002)
        if code in (0x25, 0x26, 0x27, 0x28):
            flags |= 0x0001
        event = INPUT(type=1, u=INPUTUNION(ki=KEYBDINPUT(0, scan, flags, 0, 0)))
        if user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT)) != 1:
            raise ctypes.WinError(ctypes.get_last_error())

    def click(self, x: int, y: int, button: str = "left") -> None:
        if not user32.SetCursorPos(x, y):
            raise ctypes.WinError(ctypes.get_last_error())
        down, up = (0x0002, 0x0004) if button == "left" else (0x0008, 0x0010)
        user32.mouse_event(down, 0, 0, 0, 0)
        try:
            # Roblox menus can miss a press and release delivered in the same frame.
            time.sleep(0.08)
        finally:
            user32.mouse_event(up, 0, 0, 0, 0)

    def scroll(self, x: int, y: int, amount: int) -> None:
        if not user32.SetCursorPos(x, y):
            raise ctypes.WinError(ctypes.get_last_error())
        user32.mouse_event(0x0800, 0, 0, amount & 0xFFFFFFFF, 0)

    def keyboard_listener(self, owner: Any) -> "WindowsKeyboardListener":
        return WindowsKeyboardListener(owner)


class WindowsKeyboardListener:
    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self.thread_id = None

    def run(self) -> None:
        self.thread_id = kernel32.GetCurrentThreadId()

        @HOOKPROC
        def callback(code, message, pointer):
            if code >= 0 and message in (0x100, 0x101, 0x104, 0x105):
                event = ctypes.cast(pointer, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                self.owner.handle_key(
                    event.vkCode, message in (0x100, 0x104), synthetic=bool(event.flags & 0x10)
                )
            return user32.CallNextHookEx(None, code, message, pointer)

        hook = user32.SetWindowsHookExW(13, callback, kernel32.GetModuleHandleW(None), 0)
        if not hook:
            raise ctypes.WinError(ctypes.get_last_error())
        # Create the queue before announcing readiness so close() can always post WM_QUIT.
        message = wintypes.MSG()
        user32.PeekMessageW(ctypes.byref(message), None, 0, 0, 0)
        self.owner.ready.set()
        try:
            while True:
                status = user32.GetMessageW(ctypes.byref(message), None, 0, 0)
                if status == -1:
                    raise ctypes.WinError(ctypes.get_last_error())
                if status == 0:
                    break
        finally:
            user32.UnhookWindowsHookEx(hook)

    def close(self) -> None:
        if self.thread_id is not None:
            user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)
