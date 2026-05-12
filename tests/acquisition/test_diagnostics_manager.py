"""Tests for ``DiagnosticsManager`` — the WS fan-out used by PR 2's diagnostic
envelope channel (``/api/v1/ws/diagnostics``).

Failure modes the tests cover, all of which broke the v2 implementation that
shared a single connection list with the recording-status WS:

- Per-connection lock serializes concurrent broadcasts
- A failing send_json drops the connection instead of leaving a stale entry
- A dead connection does not block other connections from receiving
- Disconnect clears the per-connection lock entry (no leak)
"""

from __future__ import annotations

import asyncio
import sys
import threading
import types

import pytest


def _install_pyspin_stubs() -> None:
    for name in ("PySpin", "simple_pyspin"):
        if name in sys.modules:
            continue
        stub = types.ModuleType(name)
        if name == "simple_pyspin":

            class _Camera:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("PySpin stub")

            stub.Camera = _Camera  # type: ignore[attr-defined]
            stub._SYSTEM = None  # type: ignore[attr-defined]
            stub.list_cameras = lambda: []  # type: ignore[attr-defined]
        sys.modules[name] = stub


_install_pyspin_stubs()


def _import_backend():
    import os
    import unittest.mock as _mock

    real_listdir = os.listdir

    def safe_listdir(path):
        if str(path).startswith("/configs"):
            return []
        return real_listdir(path)

    os.makedirs("data", exist_ok=True)
    with _mock.patch("os.listdir", side_effect=safe_listdir):
        from multi_camera.backend import fastapi as _backend_fastapi
    return _backend_fastapi


try:
    backend_fastapi = _import_backend()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Backend not importable: {exc}", allow_module_level=True)


class FakeWebSocket:
    """Minimal stand-in for starlette.websockets.WebSocket."""

    def __init__(self, name: str = "ws", fail_send: bool = False):
        self.name = name
        self.fail_send = fail_send
        self.sent: list[dict] = []
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, message: dict) -> None:
        if self.fail_send:
            raise ConnectionError(f"{self.name} dead")
        self.sent.append(message)


def _run(coro):
    return asyncio.run(coro)


class TestConnectionLifecycle:
    def test_connect_accepts_and_registers(self) -> None:
        async def body():
            mgr = backend_fastapi.DiagnosticsManager()
            ws = FakeWebSocket()
            await mgr.connect(ws)
            assert ws.accepted is True
            assert ws in mgr.active_connections
            assert ws in mgr._locks

        _run(body())

    def test_disconnect_unregisters_and_drops_lock(self) -> None:
        async def body():
            mgr = backend_fastapi.DiagnosticsManager()
            ws = FakeWebSocket()
            await mgr.connect(ws)
            mgr.disconnect(ws)
            assert ws not in mgr.active_connections
            assert ws not in mgr._locks

        _run(body())

    def test_disconnect_is_idempotent(self) -> None:
        async def body():
            mgr = backend_fastapi.DiagnosticsManager()
            ws = FakeWebSocket()
            await mgr.connect(ws)
            mgr.disconnect(ws)
            mgr.disconnect(ws)  # must not raise

        _run(body())


class TestBroadcast:
    def test_broadcast_reaches_all_connections(self) -> None:
        async def body():
            mgr = backend_fastapi.DiagnosticsManager()
            ws_a = FakeWebSocket("a")
            ws_b = FakeWebSocket("b")
            await mgr.connect(ws_a)
            await mgr.connect(ws_b)

            envelope = {
                "type": "health_report",
                "level": "ok",
                "code": "x",
                "message": "y",
            }
            await mgr.broadcast(envelope)

            assert ws_a.sent == [envelope]
            assert ws_b.sent == [envelope]

        _run(body())

    def test_broadcast_with_no_connections_is_a_noop(self) -> None:
        async def body():
            mgr = backend_fastapi.DiagnosticsManager()
            await mgr.broadcast({"type": "x"})

        _run(body())

    def test_dead_connection_is_dropped(self) -> None:
        async def body():
            mgr = backend_fastapi.DiagnosticsManager()
            live = FakeWebSocket("live")
            dead = FakeWebSocket("dead", fail_send=True)
            await mgr.connect(live)
            await mgr.connect(dead)

            await mgr.broadcast(
                {"type": "x", "level": "ok", "code": "k", "message": "m"}
            )

            assert dead not in mgr.active_connections
            assert dead not in mgr._locks
            assert live in mgr.active_connections
            assert len(live.sent) == 1

        _run(body())

    def test_per_connection_lock_serializes_concurrent_broadcasts(self) -> None:
        """Two concurrent broadcast() calls to the same connection must not interleave.

        The v2 bug surfaced as ``AssertionError in _drain_helper`` when the
        websockets library's keepalive ping coroutine contended with a user-code
        broadcast on the same writer. Per-connection asyncio.Lock prevents the
        contention from racing into send_json.
        """

        async def body():
            mgr = backend_fastapi.DiagnosticsManager()

            in_flight = 0
            max_in_flight = 0
            order: list[str] = []

            class TrackingWS(FakeWebSocket):
                async def send_json(self, message):
                    nonlocal in_flight, max_in_flight
                    in_flight += 1
                    max_in_flight = max(max_in_flight, in_flight)
                    order.append(f"start:{message['code']}")
                    await asyncio.sleep(0)
                    order.append(f"end:{message['code']}")
                    in_flight -= 1
                    self.sent.append(message)

            ws = TrackingWS()
            await mgr.connect(ws)

            await asyncio.gather(
                mgr.broadcast(
                    {"type": "x", "level": "ok", "code": "A", "message": "m"}
                ),
                mgr.broadcast(
                    {"type": "x", "level": "ok", "code": "B", "message": "m"}
                ),
                mgr.broadcast(
                    {"type": "x", "level": "ok", "code": "C", "message": "m"}
                ),
            )

            assert max_in_flight == 1, (
                f"expected serial sends, observed {max_in_flight} concurrent"
            )
            for i in range(0, len(order), 2):
                assert order[i].startswith("start:")
                assert order[i + 1].startswith("end:")
                assert order[i].split(":")[1] == order[i + 1].split(":")[1]

        _run(body())


class TestBroadcastEvent:
    def test_broadcast_event_from_loop_thread(self) -> None:
        """Calling broadcast_event from inside the asyncio loop should schedule a task."""

        async def body():
            captured: list[dict] = []

            async def fake_broadcast(message):
                captured.append(message)

            old = backend_fastapi.diagnostics_manager.broadcast
            backend_fastapi.diagnostics_manager.broadcast = fake_broadcast
            try:
                backend_fastapi.broadcast_event(
                    event_type="health_report",
                    level="error",
                    code="dhcp_service_down",
                    message="The camera IP server is not running.",
                    details={"systemctl": "inactive"},
                )
                await asyncio.sleep(0.05)
            finally:
                backend_fastapi.diagnostics_manager.broadcast = old

            assert len(captured) == 1
            env = captured[0]
            assert env["type"] == "health_report"
            assert env["level"] == "error"
            assert env["code"] == "dhcp_service_down"
            assert env["details"] == {"systemctl": "inactive"}
            assert "ts" in env and env["ts"].endswith("Z")

        _run(body())

    def test_broadcast_event_from_non_asyncio_thread(self) -> None:
        """From a non-asyncio thread the helper must use run_coroutine_threadsafe.

        Simulates HealthIdlePoller's daemon thread or FlirRecorder callbacks.
        """

        captured: list[dict] = []
        done = threading.Event()

        async def fake_broadcast(message):
            captured.append(message)
            done.set()

        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        loop_thread.start()

        old_loop = backend_fastapi.loop
        old_broadcast = backend_fastapi.diagnostics_manager.broadcast
        backend_fastapi.loop = loop
        backend_fastapi.diagnostics_manager.broadcast = fake_broadcast

        try:
            def caller():
                backend_fastapi.broadcast_event(
                    event_type="trial_failed",
                    level="error",
                    code="trial_failed",
                    message="Recording failed: TimeoutError",
                )

            t = threading.Thread(target=caller)
            t.start()
            t.join()

            assert done.wait(timeout=2.0), "broadcast never landed on the asyncio loop"
        finally:
            backend_fastapi.diagnostics_manager.broadcast = old_broadcast
            backend_fastapi.loop = old_loop
            loop.call_soon_threadsafe(loop.stop)
            loop_thread.join(timeout=1.0)

        assert len(captured) == 1
        assert captured[0]["code"] == "trial_failed"
