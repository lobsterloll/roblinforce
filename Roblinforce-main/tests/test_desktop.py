"""Desktop safety contracts tested without touching the user's desktop."""

import pytest

from roblinforce import desktop


class FakeBackend:
    def __init__(self):
        self.focused = True
        self.area = {"left": 10, "top": 20, "width": 640, "height": 480}
        self.events = []
        self.clicks = []

    def is_game_focused(self, window):
        return self.focused

    def client_capture_area(self, window):
        return self.area

    def send_key(self, key, down):
        self.events.append((key, down))

    def click(self, x, y, button="left"):
        self.clicks.append((x, y))


@pytest.fixture
def backend(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(desktop, "_backend", backend)
    return backend


def test_keys_are_logical_and_independent_of_platform():
    assert desktop.virtual_key("w") == ord("W")
    assert desktop.virtual_key("F8") == 0x77
    assert desktop.virtual_key("SPACE") == 0x20
    with pytest.raises(ValueError):
        desktop.virtual_key("command")


def test_output_releases_previous_key_before_pressing_next(backend):
    output = desktop.KeyOutput(42)
    output.set_pressed({65})
    output.set_pressed({68})
    assert backend.events == [(65, True), (65, False), (68, True)]
    output.release_all()
    assert output.pressed == set()


def test_output_releases_and_refuses_new_keys_after_focus_loss(backend):
    output = desktop.KeyOutput(42)
    output.set_pressed({65})
    backend.focused = False
    output.set_pressed({68})
    assert backend.events == [(65, True), (65, False)]


def test_direct_send_also_checks_focus(backend):
    output = desktop.KeyOutput(42)
    backend.focused = False
    output.send(65, True)
    assert backend.events == []


def test_release_attempts_every_key_when_one_release_fails(backend):
    output = desktop.KeyOutput(42)
    output.set_pressed({65, 68})
    original_send = backend.send_key

    def send_key(key, down):
        if key == 65 and not down:
            raise OSError("failed release")
        original_send(key, down)

    backend.send_key = send_key
    with pytest.raises(OSError, match="failed release"):
        output.release_all()
    assert (68, False) in backend.events
    assert output.pressed == {65}


def test_physical_keys_ignore_synthetic_events_and_repeated_downs():
    codes = {
        "accelerate": 87,
        "brake": 83,
        "handbrake": 32,
        "left": 65,
        "right": 68,
        "toggle": 119,
        "stop": 120,
    }
    keys = desktop.PhysicalKeys(codes)
    keys.handle_key(119, True, synthetic=True)
    assert not keys.toggle.is_set()
    keys.handle_key(119, True)
    assert keys.toggle.is_set()
    keys.toggle.clear()
    keys.handle_key(119, True)
    assert not keys.toggle.is_set()
    keys.handle_key(119, False)
    keys.handle_key(119, True)
    assert keys.toggle.is_set()
    keys.handle_key(87, True)
    assert keys.takeover.is_set()
    keys.handle_key(120, True)
    assert keys.stop.is_set()


def test_physical_escape_takes_over_even_with_stop_disabled():
    codes = {
        "accelerate": 87,
        "brake": 83,
        "handbrake": 32,
        "left": 65,
        "right": 68,
        "toggle": 119,
        "stop": 120,
    }
    keys = desktop.PhysicalKeys(codes, stop_enabled=False)
    keys.handle_key(120, True)
    assert not keys.stop.is_set()
    keys.handle_key(0x1B, True)
    assert keys.takeover.is_set()


def test_click_requires_focus_current_bounds_and_inside_point(backend):
    area = backend.area.copy()
    assert desktop.click_game_point(42, area, (50, 60))
    assert backend.clicks == [(60, 80)]
    assert not desktop.click_game_point(42, area, (640, 0))
    backend.area = {**area, "left": 30}
    assert not desktop.click_game_point(42, area, (50, 60))
    backend.area = area
    backend.focused = False
    assert not desktop.click_game_point(42, area, (50, 60))
    assert len(backend.clicks) == 1


def test_unsupported_os_has_actionable_error(monkeypatch):
    monkeypatch.setattr(desktop.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="Windows and macOS"):
        desktop._create_backend()


def test_capture_normalizes_retina_size_and_preserves_bgr():
    from types import SimpleNamespace
    import numpy as np

    screenshot = np.full((960, 1280, 4), 80, np.uint8)
    capture = SimpleNamespace(grab=lambda area: screenshot)
    frame = desktop.capture_frame(capture, {"width": 640, "height": 480})
    assert frame.shape == (480, 640, 3)
    assert np.all(frame == 80)
    frame[:] = 0
    assert np.all(screenshot == 80)


def test_scroll_and_click_button_validation(backend):
    from unittest.mock import Mock

    backend.scroll = Mock()
    assert desktop.scroll_game_point(42, backend.area, (50, 60), -120)
    backend.scroll.assert_called_once_with(60, 80, -120)
    assert not desktop.scroll_game_point(42, backend.area, (-1, 0), 120)
    with pytest.raises(ValueError, match="button"):
        desktop.click_game_point(42, backend.area, (50, 60), "middle")


def test_backend_is_lazy_cached_and_dispatches(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    backend = SimpleNamespace(
        initialize=Mock(),
        find_game_window=Mock(return_value=42),
        physical_down=Mock(return_value=True),
    )
    factory = Mock(return_value=backend)
    monkeypatch.setattr(desktop, "_backend", None)
    monkeypatch.setattr(desktop, "_create_backend", factory)
    desktop.initialize_desktop()
    assert desktop.find_game_window("Roblox") == 42
    assert desktop.physical_down(87)
    factory.assert_called_once()
    backend.initialize.assert_called_once()


def test_native_factories_are_lazy(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import sys

    for platform, module, constructor in (
        ("win32", "desktop_windows", "WindowsDesktop"),
        ("darwin", "desktop_macos", "MacDesktop"),
    ):
        factory = Mock(return_value=object())
        monkeypatch.setitem(
            sys.modules, f"roblinforce.{module}", SimpleNamespace(**{constructor: factory})
        )
        monkeypatch.setattr(desktop.sys, "platform", platform)
        assert desktop._create_backend() is factory.return_value


def test_listener_unexpected_exit_stops_driving(backend):
    from types import SimpleNamespace

    codes = {
        "accelerate": 87,
        "brake": 83,
        "handbrake": 32,
        "left": 65,
        "right": 68,
        "toggle": 119,
        "stop": 120,
    }
    backend.keyboard_listener = lambda owner: SimpleNamespace(run=lambda: None, close=lambda: None)
    keys = desktop.PhysicalKeys(codes)
    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        keys.start()
    assert keys.stop.is_set()
    keys.close()


def test_listener_orderly_start_and_close(backend):
    from types import SimpleNamespace
    import threading

    stopped = threading.Event()
    codes = {
        "accelerate": 87,
        "brake": 83,
        "handbrake": 32,
        "left": 65,
        "right": 68,
        "toggle": 119,
        "stop": 120,
    }

    def listener(owner):
        def run():
            owner.ready.set()
            stopped.wait(1)

        return SimpleNamespace(run=run, close=stopped.set)

    backend.keyboard_listener = listener
    keys = desktop.PhysicalKeys(codes)
    keys.start()
    keys.check_health()
    keys.close()
    assert keys.error is None
