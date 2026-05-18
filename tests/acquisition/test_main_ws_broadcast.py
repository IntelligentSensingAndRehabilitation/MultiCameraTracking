"""Tests for ``ConnectionManager``, ``receive_status``, and ``lifespan``'s
loop binding on the ``/api/v1/ws`` broadcast path."""

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
    """Minimal stand-in for ``starlette.websockets.WebSocket``."""

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


class TestConnectionManagerLifecycle:
    def test_connect_accepts_and_registers_with_lock(self) -> None:
        async def body():
            mgr = backend_fastapi.ConnectionManager()
            ws = FakeWebSocket()
            await mgr.connect(ws)
            assert ws.accepted is True
            assert ws in mgr.active_connections
            assert ws in mgr._locks

        _run(body())

    def test_disconnect_unregisters_and_drops_lock(self) -> None:
        async def body():
            mgr = backend_fastapi.ConnectionManager()
            ws = FakeWebSocket()
            await mgr.connect(ws)
            mgr.disconnect(ws)
            assert ws not in mgr.active_connections
            assert ws not in mgr._locks

        _run(body())

    def test_disconnect_is_idempotent(self) -> None:
        async def body():
            mgr = backend_fastapi.ConnectionManager()
            ws = FakeWebSocket()
            await mgr.connect(ws)
            mgr.disconnect(ws)
            mgr.disconnect(ws)  # must not raise

        _run(body())


class TestConnectionManagerBroadcast:
    def test_broadcast_reaches_all_connections(self) -> None:
        async def body():
            mgr = backend_fastapi.ConnectionManager()
            ws_a = FakeWebSocket("a")
            ws_b = FakeWebSocket("b")
            await mgr.connect(ws_a)
            await mgr.connect(ws_b)

            update = {"status": "Idle"}
            await mgr.broadcast(update)

            assert ws_a.sent == [update]
            assert ws_b.sent == [update]

        _run(body())

    def test_broadcast_with_no_connections_is_a_noop(self) -> None:
        async def body():
            mgr = backend_fastapi.ConnectionManager()
            await mgr.broadcast({"status": "Idle"})

        _run(body())

    def test_dead_connection_is_dropped(self) -> None:
        async def body():
            mgr = backend_fastapi.ConnectionManager()
            live = FakeWebSocket("live")
            dead = FakeWebSocket("dead", fail_send=True)
            await mgr.connect(live)
            await mgr.connect(dead)

            await mgr.broadcast({"status": "Idle"})

            assert dead not in mgr.active_connections
            assert dead not in mgr._locks
            assert live in mgr.active_connections
            assert len(live.sent) == 1

        _run(body())

    def test_per_connection_lock_serializes_concurrent_broadcasts(self) -> None:
        """Concurrent broadcasts to the same connection must not interleave."""

        async def body():
            mgr = backend_fastapi.ConnectionManager()

            in_flight = 0
            max_in_flight = 0

            class CountingWS(FakeWebSocket):
                async def send_json(self, message: dict) -> None:
                    nonlocal in_flight, max_in_flight
                    in_flight += 1
                    max_in_flight = max(max_in_flight, in_flight)
                    await asyncio.sleep(0.01)
                    in_flight -= 1
                    self.sent.append(message)

            ws = CountingWS("counted")
            await mgr.connect(ws)

            await asyncio.gather(
                mgr.broadcast({"status": "A"}),
                mgr.broadcast({"status": "B"}),
                mgr.broadcast({"status": "C"}),
            )

            assert max_in_flight == 1, (
                f"Expected serialized writes (max_in_flight=1), got {max_in_flight}"
            )
            assert len(ws.sent) == 3

        _run(body())


class TestReceiveStatusCrossThread:
    """``receive_status`` must broadcast whether invoked from the loop thread
    or a worker thread."""

    def test_status_callback_from_worker_thread_reaches_main_ws(self) -> None:
        async def body():
            backend_fastapi.loop = asyncio.get_running_loop()

            ws = FakeWebSocket("recv-status")
            await backend_fastapi.manager.connect(ws)

            try:
                done = threading.Event()

                def fire_from_thread():
                    backend_fastapi.receive_status("Synchronizing")
                    done.set()

                threading.Thread(target=fire_from_thread, daemon=True).start()

                assert done.wait(timeout=2.0), "receive_status did not return"
                for _ in range(20):
                    if ws.sent:
                        break
                    await asyncio.sleep(0.05)

                assert ws.sent == [{"status": "Synchronizing"}]
            finally:
                backend_fastapi.manager.disconnect(ws)

        _run(body())

    def test_status_callback_from_loop_thread_reaches_main_ws(self) -> None:
        async def body():
            backend_fastapi.loop = asyncio.get_running_loop()

            ws = FakeWebSocket("recv-status-loop")
            await backend_fastapi.manager.connect(ws)

            try:
                backend_fastapi.receive_status("Idle")
                for _ in range(20):
                    if ws.sent:
                        break
                    await asyncio.sleep(0.05)

                assert ws.sent == [{"status": "Idle"}]
            finally:
                backend_fastapi.manager.disconnect(ws)

        _run(body())

    def test_progress_update_includes_progress_field(self) -> None:
        async def body():
            backend_fastapi.loop = asyncio.get_running_loop()

            ws = FakeWebSocket("recv-progress")
            await backend_fastapi.manager.connect(ws)

            try:
                backend_fastapi.receive_status("Recording", progress=0.42)
                for _ in range(20):
                    if ws.sent:
                        break
                    await asyncio.sleep(0.05)

                assert ws.sent == [{"status": "Recording", "progress": 0.42}]
            finally:
                backend_fastapi.manager.disconnect(ws)

        _run(body())


class TestLifespanCapturesRunningLoop:
    """``lifespan()`` must rebind module-level ``loop`` to the running loop so
    ``run_coroutine_threadsafe`` from worker threads targets uvicorn's loop."""

    def test_lifespan_rebinds_loop_to_running_loop(self) -> None:
        import unittest.mock as _mock
        from contextlib import ExitStack

        from fastapi import FastAPI

        backend_fastapi.loop = None

        async def body():
            with ExitStack() as stack:
                stack.enter_context(
                    _mock.patch.object(backend_fastapi, "FlirRecorder", autospec=True)
                )
                stack.enter_context(
                    _mock.patch.object(
                        backend_fastapi, "HealthIdlePoller", autospec=True
                    )
                )
                stack.enter_context(
                    _mock.patch.object(
                        backend_fastapi, "get_db", return_value=_mock.MagicMock()
                    )
                )
                stack.enter_context(
                    _mock.patch.object(backend_fastapi, "synchronize_to_datajoint")
                )
                async with backend_fastapi.lifespan(FastAPI()):
                    assert backend_fastapi.loop is asyncio.get_running_loop(), (
                        "lifespan() did not re-bind module-level loop to the "
                        "running loop; worker-thread broadcasts via "
                        "run_coroutine_threadsafe will silently drop."
                    )

        _run(body())
