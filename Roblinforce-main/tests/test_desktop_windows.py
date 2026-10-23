"""Win32 ABI calls are fakes even when these tests run on macOS or Linux."""

import ctypes
import importlib
import sys
from unittest.mock import Mock

import pytest


class FakeDLL:
    def __init__(self):
        self.functions = {}

    def __getattr__(self, name):
        if name not in self.functions:
            self.functions[name] = Mock(return_value=1)
        return self.functions[name]


@pytest.fixture
def native(monkeypatch):
    user32, kernel32 = FakeDLL(), FakeDLL()
    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda name, **kwargs: user32 if name == "user32" else kernel32,
        raising=False,
    )
    monkeypatch.setattr(ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE, raising=False)
    monkeypatch.setattr(ctypes, "WinError", lambda *args: OSError("native failure"), raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: 5, raising=False)
    sys.modules.pop("roblinforce.desktop_windows", None)
    module = importlib.import_module("roblinforce.desktop_windows")
    yield module, user32, kernel32
    sys.modules.pop("roblinforce.desktop_windows", None)


def test_window_search_capture_bounds_and_focus(native):
    module, user32, _ = native
    backend = module.WindowsDesktop()
    backend.initialize()
    user32.GetWindowTextLengthW.return_value = 6

    def title(window, buffer, size):
        buffer.value = "Roblox"

    user32.GetWindowTextW.side_effect = title
    user32.EnumWindows.side_effect = lambda callback, context: callback(42, 0)
    assert backend.find_game_window("roblox") == 42
    user32.IsWindowVisible.return_value = 0
    assert backend.find_game_window("Roblox") is None
    user32.IsWindowVisible.return_value = 1
    user32.IsIconic.return_value = 0

    def bounds(window, pointer):
        pointer._obj.right, pointer._obj.bottom = 640, 480
        return 1

    def origin(window, pointer):
        pointer._obj.x, pointer._obj.y = 10, 20
        return 1

    user32.GetClientRect.side_effect = bounds
    user32.ClientToScreen.side_effect = origin
    assert backend.client_capture_area(42) == {"left": 10, "top": 20, "width": 640, "height": 480}
    user32.GetForegroundWindow.return_value = 42
    assert backend.is_game_focused(42)
    user32.IsIconic.return_value = 1
    assert backend.client_capture_area(42) is None
    user32.IsIconic.return_value = 0
    user32.GetClientRect.side_effect = lambda *args: 0
    assert backend.client_capture_area(42) is None
    user32.GetClientRect.side_effect = bounds
    user32.ClientToScreen.side_effect = lambda *args: 0
    assert backend.client_capture_area(42) is None
    assert backend.client_capture_area(None) is None


def test_keyboard_mouse_and_native_errors(native):
    module, user32, _ = native
    backend = module.WindowsDesktop()
    backend.send_key(65, True)
    backend.send_key(0x25, False)
    assert user32.SendInput.call_count == 2
    user32.GetAsyncKeyState.return_value = 0x8000
    assert backend.physical_down(65)
    backend.click(10, 20)
    backend.click(10, 20, button="right")
    backend.scroll(10, 20, -120)
    assert user32.mouse_event.call_count == 5
    user32.SetCursorPos.return_value = 0
    with pytest.raises(OSError):
        backend.click(10, 20)
    with pytest.raises(OSError):
        backend.scroll(10, 20, 120)
    user32.SendInput.return_value = 0
    with pytest.raises(OSError):
        backend.send_key(65, True)


def test_native_listener_routes_events_and_unhooks(native):
    module, user32, _ = native
    owner = Mock()
    backend = module.WindowsDesktop()
    listener = backend.keyboard_listener(owner)

    def install(kind, callback, instance, thread_id):
        event = module.KBDLLHOOKSTRUCT(vkCode=87, flags=0)
        callback(0, 0x100, ctypes.addressof(event))
        event.flags = 0x10
        callback(0, 0x101, ctypes.addressof(event))
        callback(-1, 0, 0)
        return 99

    user32.SetWindowsHookExW.side_effect = install
    user32.GetMessageW.return_value = 0
    listener.run()
    assert owner.handle_key.call_args_list[0].args == (87, True)
    assert owner.handle_key.call_args_list[1].kwargs == {"synthetic": True}
    owner.ready.set.assert_called_once()
    user32.UnhookWindowsHookEx.assert_called_once_with(99)
    listener.close()
    user32.PostThreadMessageW.assert_called_once()


def test_native_listener_failures_are_not_swallowed(native):
    module, user32, _ = native
    listener = module.WindowsKeyboardListener(Mock())
    user32.SetWindowsHookExW.return_value = 0
    with pytest.raises(OSError):
        listener.run()
    user32.SetWindowsHookExW.return_value = 99
    user32.GetMessageW.return_value = -1
    with pytest.raises(OSError):
        listener.run()
    user32.UnhookWindowsHookEx.assert_called_once_with(99)


def test_click_has_a_button_dwell_and_releases_after_delay_error(native, monkeypatch):
    module, user32, _ = native
    backend = module.WindowsDesktop()
    sleeps = Mock(side_effect=OSError("interrupted delay"))
    monkeypatch.setattr("time.sleep", sleeps)
    with pytest.raises(OSError, match="interrupted delay"):
        backend.click(10, 20)
    sleeps.assert_called_once_with(0.08)
    assert user32.mouse_event.call_args_list[-1].args[0] == 0x0004
