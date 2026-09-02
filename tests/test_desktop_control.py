from __future__ import annotations

import os
import queue
import socket
from types import SimpleNamespace

import pytest

from desktop_app.darwin_neg_control import (
    DarwinControlApp,
    listening_pid,
    process_snapshot,
    same_process,
)


@pytest.mark.skipif(os.name != "nt", reason="Windows controller process identity")
def test_process_identity_rejects_pid_reuse_and_image_changes() -> None:
    snapshot = process_snapshot(os.getpid())
    assert snapshot is not None
    assert same_process(dict(snapshot), snapshot)

    stale = dict(snapshot)
    stale["created"] += 1
    assert not same_process(stale, snapshot)

    wrong_image = dict(snapshot)
    wrong_image["image"] = str(snapshot["image"]) + ".other"
    assert not same_process(wrong_image, snapshot)


@pytest.mark.skipif(os.name != "nt", reason="Windows controller listener discovery")
def test_listening_pid_resolves_current_process() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert listening_pid(port) == os.getpid()


def test_close_destroys_window_after_stopped_event() -> None:
    class FakeRoot:
        destroyed = False
        after_called = False

        def destroy(self) -> None:
            self.destroyed = True

        def after(self, _delay: int, _callback: object) -> None:
            self.after_called = True

    app = DarwinControlApp.__new__(DarwinControlApp)
    app.events = queue.Queue()
    app.events.put(("stopped", None))
    app.root = FakeRoot()
    app.closing = True
    app.external_stack = True
    app.running_since = 1.0
    app.uptime_var = SimpleNamespace(set=lambda _value: None)
    app.pid_var = SimpleNamespace(set=lambda _value: None)
    app._set_status = lambda *_args: None
    app._set_buttons = lambda **_kwargs: None

    app._drain_events()

    assert app.root.destroyed
    assert not app.root.after_called
