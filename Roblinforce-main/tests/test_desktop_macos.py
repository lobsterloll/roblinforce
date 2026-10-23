"""Quartz is replaced completely so tests never request permissions or send input."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from roblinforce.desktop import PhysicalKeys
from roblinforce.desktop_macos import MacDesktop, MacKeyboardListener, SYNTHETIC_EVENT_TAG


@pytest.fixture
def native(monkeypatch):
    names = (
        "kCGWindowListOptionOnScreenOnly",
        "kCGWindowListExcludeDesktopElements",
        "kCGNullWindowID",
        "kCGEventSourceStateHIDSystemState",
        "kCGEventSourceUserData",
        "kCGHIDEventTap",
        "kCGEventLeftMouseDown",
        "kCGEventLeftMouseUp",
        "kCGMouseButtonLeft",
        "kCGEventRightMouseDown",
        "kCGEventRightMouseUp",
        "kCGMouseButtonRight",
        "kCGEventMouseMoved",
        "kCGScrollEventUnitLine",
        "kCGEventKeyDown",
        "kCGEventKeyUp",
        "kCGEventFlagsChanged",
        "kCGEventTapDisabledByTimeout",
        "kCGEventTapDisabledByUserInput",
        "kCGEventSourceUnixProcessID",
        "kCGKeyboardEventKeycode",
        "kCGSessionEventTap",
        "kCGHeadInsertEventTap",
        "kCGEventTapOptionListenOnly",
        "kCFRunLoopCommonModes",
        "kCFRunLoopDefaultMode",
    )
    quartz = SimpleNamespace(**{name: index for index, name in enumerate(names, 1)})
    for name in (
        "kCGWindowLayer",
        "kCGWindowName",
        "kCGWindowOwnerName",
        "kCGWindowNumber",
        "kCGWindowBounds",
        "kCGWindowOwnerPID",
    ):
        setattr(quartz, name, name)
    window = {
        quartz.kCGWindowNumber: 42,
        quartz.kCGWindowName: "Game",
        quartz.kCGWindowOwnerName: "Roblox",
        quartz.kCGWindowOwnerPID: 7,
        quartz.kCGWindowBounds: {"X": 10, "Y": 20, "Width": 640, "Height": 480},
    }
    quartz.CGWindowListCopyWindowInfo = Mock(return_value=[window])
    for name in (
        "CGPreflightScreenCaptureAccess",
        "CGPreflightPostEventAccess",
        "CGPreflightListenEventAccess",
        "CGEventSourceKeyState",
    ):
        setattr(quartz, name, Mock(return_value=True))
    for name in (
        "CGEventSetIntegerValueField",
        "CGEventPost",
        "CGEventTapEnable",
        "CFRunLoopAddSource",
        "CFRunLoopRemoveSource",
        "CFMachPortInvalidate",
        "CFRunLoopStop",
    ):
        setattr(quartz, name, Mock())
    for name in (
        "CGEventCreateKeyboardEvent",
        "CGEventCreateMouseEvent",
        "CGEventCreateScrollWheelEvent",
        "CGEventTapCreate",
        "CFMachPortCreateRunLoopSource",
        "CFRunLoopGetCurrent",
    ):
        setattr(quartz, name, Mock(return_value=object()))
    quartz.CGEventGetIntegerValueField = lambda event, field: event.get(field, 0)
    appkit = SimpleNamespace(
        NSWorkspace=SimpleNamespace(
            sharedWorkspace=lambda: SimpleNamespace(
                frontmostApplication=lambda: SimpleNamespace(processIdentifier=lambda: 7)
            )
        )
    )
    monkeypatch.setitem(sys.modules, "Quartz", quartz)
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    return quartz, appkit, window


def test_permissions_and_window_focus(native):
    quartz, appkit, window = native
    backend = MacDesktop()
    backend.initialize()
    assert backend.find_game_window("roblox") == 42
    assert backend.find_game_window("other") is None
    assert backend.client_capture_area(42) == {"left": 10, "top": 20, "width": 640, "height": 480}
    assert backend.client_capture_area(43) is None
    assert backend.is_game_focused(42)
    assert not backend.is_game_focused(43)
    window[quartz.kCGWindowBounds]["Width"] = 10
    assert backend.client_capture_area(42) is None
    quartz.CGPreflightListenEventAccess.return_value = False
    with pytest.raises(RuntimeError, match="Input Monitoring"):
        backend.initialize()
    quartz.CGEventPost.assert_not_called()


def test_frontmost_application_must_match_selected_window(native):
    quartz, appkit, _ = native
    backend = MacDesktop()
    appkit.NSWorkspace.sharedWorkspace = lambda: SimpleNamespace(frontmostApplication=lambda: None)
    assert not backend.is_game_focused(42)
    quartz.CGWindowListCopyWindowInfo.return_value = []
    assert not backend.is_game_focused(42)


def test_tagged_keyboard_pointer_and_scroll_events(native):
    quartz, _, _ = native
    backend = MacDesktop()
    backend.send_key(87, True)
    quartz.CGEventCreateKeyboardEvent.assert_called_with(None, 13, True)
    assert backend.physical_down(87)
    backend.click(10, 20)
    backend.click(10, 20, button="right")
    backend.scroll(10, 20, -240)
    quartz.CGEventCreateScrollWheelEvent.assert_called_with(
        None, quartz.kCGScrollEventUnitLine, 1, -2
    )
    backend.scroll(10, 20, 1)
    assert quartz.CGEventPost.call_count == 9
    assert all(
        call.args[2] == SYNTHETIC_EVENT_TAG
        for call in quartz.CGEventSetIntegerValueField.call_args_list
    )
    with pytest.raises(OSError, match="create"):
        backend._post_event(None)


def test_event_tap_routes_keys_filters_tag_and_stops_on_disable(native):
    quartz, _, _ = native
    owner = PhysicalKeys(
        {
            "accelerate": 87,
            "brake": 83,
            "handbrake": 32,
            "left": 65,
            "right": 68,
            "toggle": 119,
            "stop": 120,
        }
    )
    listener = MacDesktop().keyboard_listener(owner)

    def dispatch(*_args):
        event = {quartz.kCGKeyboardEventKeycode: 100}
        listener.callback(
            None,
            quartz.kCGEventKeyDown,
            {**event, quartz.kCGEventSourceUserData: SYNTHETIC_EVENT_TAG},
            None,
        )
        assert not owner.toggle.is_set()
        listener.callback(None, quartz.kCGEventKeyDown, event, None)
        assert owner.toggle.is_set()
        listener.callback(None, quartz.kCGEventKeyUp, event, None)
        listener.callback(
            None, quartz.kCGEventFlagsChanged, {quartz.kCGKeyboardEventKeycode: 56}, None
        )
        listener.callback(None, quartz.kCGEventKeyDown, {quartz.kCGKeyboardEventKeycode: 500}, None)
        listener.callback(None, quartz.kCGEventTapDisabledByTimeout, event, None)

    quartz.CFRunLoopRunInMode = dispatch
    listener.run()
    assert owner.ready.is_set() and owner.stop.is_set()
    assert "disabled" in str(owner.error)
    listener.close()
    quartz.CFRunLoopStop.assert_called_once()
    quartz.CFMachPortInvalidate.assert_called_once()


def test_missing_tap_fails_closed(native):
    quartz, _, _ = native
    quartz.CGEventTapCreate.return_value = None
    listener = MacKeyboardListener(quartz, SimpleNamespace())
    with pytest.raises(RuntimeError, match="permissions"):
        listener.run()


def test_missing_dependency_has_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "Quartz", None)
    with pytest.raises(RuntimeError, match="requirements.txt"):
        MacDesktop()


def test_respawn_confirmation_uses_mac_return_key(native):
    quartz, _, _ = native
    backend = MacDesktop()
    backend.send_key(0x0D, True)
    quartz.CGEventCreateKeyboardEvent.assert_called_with(None, 36, True)


def test_click_has_a_button_dwell_and_releases_after_delay_error(native, monkeypatch):
    quartz, _, _ = native
    backend = MacDesktop()
    sleeps = Mock(side_effect=OSError("interrupted delay"))
    monkeypatch.setattr("time.sleep", sleeps)
    with pytest.raises(OSError, match="interrupted delay"):
        backend.click(10, 20)
    sleeps.assert_called_once_with(0.08)
    assert quartz.CGEventCreateMouseEvent.call_args_list[-1].args[1] == quartz.kCGEventLeftMouseUp


def test_generic_shift_binding_accepts_either_physical_shift(native):
    quartz, _, _ = native
    quartz.CGEventSourceKeyState.side_effect = lambda source, code: code == 60
    assert MacDesktop().physical_down(0x10)
