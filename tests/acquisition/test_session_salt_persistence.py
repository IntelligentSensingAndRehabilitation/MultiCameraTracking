"""Tests for the session-scoped config_hash_salt sidecar.

Verifies the two halves of the lifecycle:
  - POST /rig/recalibrate writes the new salt to
    {session.recording_path}/.config_hash_salt.
  - set_session restores the salt from that sidecar onto the
    FlirRecorder so trials after a mid-session restart land under the
    same camera_config_hash as trials before the restart.
"""

from __future__ import annotations

import asyncio
import datetime
import sys
import types
import unittest.mock as mock
from pathlib import Path

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

    real_listdir = os.listdir

    def safe_listdir(path):
        if str(path).startswith("/configs"):
            return []
        return real_listdir(path)

    os.makedirs("data", exist_ok=True)
    with mock.patch("os.listdir", side_effect=safe_listdir):
        from multi_camera.backend import fastapi as _backend_fastapi
    return _backend_fastapi


try:
    backend_fastapi = _import_backend()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Backend not importable: {exc}", allow_module_level=True)


def _make_session(tmp_path: Path):
    """Build a Session pointed at a real on-disk directory under tmp_path
    so the sidecar helpers can actually write/read."""
    session_dir = tmp_path / "t111" / "20260520"
    session_dir.mkdir(parents=True)
    return backend_fastapi.Session(
        participant_name="t111",
        session_date=datetime.date(2026, 5, 20),
        recording_path=str(session_dir),
    )


class TestSidecarRoundTrip:
    """The two helpers are the source of truth for the sidecar contract;
    the endpoint wiring just calls them. Round-trip them directly."""

    def test_persist_then_load(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        backend_fastapi._persist_session_salt(session, "deadbeefdeadbeef")
        assert backend_fastapi._load_session_salt(session) == "deadbeefdeadbeef"

    def test_load_returns_empty_when_no_sidecar(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        assert backend_fastapi._load_session_salt(session) == ""

    def test_helpers_noop_when_session_is_none(self, tmp_path: Path) -> None:
        # Shouldn't raise; persist silently skips, load returns "".
        backend_fastapi._persist_session_salt(None, "x")
        assert backend_fastapi._load_session_salt(None) == ""


class TestEndpointPersistsSalt:
    """POST /rig/recalibrate writes the salt to the session sidecar."""

    def _make_state(self, session, recorder):
        state = backend_fastapi.GlobalState()
        state.current_session = session
        state.acquisition = recorder
        state.recording_status = "Idle"
        return state

    def test_recalibrate_writes_sidecar(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        recorder = mock.MagicMock()
        recorder.bump_config_hash = mock.MagicMock(return_value="abc123")
        # get_salt is called AFTER bump_config_hash to grab what was rotated.
        recorder.get_salt = mock.MagicMock(return_value="newsalt123")

        state = self._make_state(session, recorder)

        async def body():
            with mock.patch.object(
                backend_fastapi, "get_global_state", return_value=state
            ), mock.patch.object(
                backend_fastapi, "broadcast_event", new=mock.MagicMock()
            ):
                return await backend_fastapi.mark_rig_recalibrate()

        result = asyncio.run(body())
        assert result == {"status": "success", "new_config_hash": "abc123"}

        sidecar = Path(session.recording_path) / ".config_hash_salt"
        assert sidecar.exists()
        assert sidecar.read_text() == "newsalt123"

    def test_recalibrate_without_active_session_does_not_crash(
        self, tmp_path: Path
    ) -> None:
        """If the operator clicks 'Camera moved' before opening a session
        (rare but possible), the bump still happens — there's just nowhere
        to persist it. Endpoint must not error.
        """
        recorder = mock.MagicMock()
        recorder.bump_config_hash = mock.MagicMock(return_value="abc123")
        recorder.get_salt = mock.MagicMock(return_value="newsalt123")

        state = backend_fastapi.GlobalState()
        state.current_session = None
        state.acquisition = recorder
        state.recording_status = "Idle"

        async def body():
            with mock.patch.object(
                backend_fastapi, "get_global_state", return_value=state
            ), mock.patch.object(
                backend_fastapi, "broadcast_event", new=mock.MagicMock()
            ):
                return await backend_fastapi.mark_rig_recalibrate()

        result = asyncio.run(body())
        assert result["status"] == "success"


class TestSetSessionRestoresSalt:
    """Mid-session restart path: operator reopens the same participant +
    date, set_session reads the sidecar and seeds the recorder's salt."""

    def test_set_session_restores_existing_salt(self, tmp_path: Path) -> None:
        session = _make_session(tmp_path)
        # Simulate a prior bump that wrote a sidecar before the restart.
        (Path(session.recording_path) / ".config_hash_salt").write_text("priorsalt")

        recorder = mock.MagicMock()
        state = backend_fastapi.GlobalState()
        state.acquisition = recorder

        # Stub the set_session pieces that touch the filesystem / DB.
        with mock.patch.object(
            backend_fastapi, "get_global_state", return_value=state
        ), mock.patch.object(
            backend_fastapi, "RECORDING_BASE", str(tmp_path)
        ), mock.patch.object(
            backend_fastapi.datetime,
            "date",
            mock.MagicMock(today=mock.MagicMock(return_value=session.session_date)),
        ):
            asyncio.run(backend_fastapi.set_session(subject_id="t111", db=mock.MagicMock()))

        recorder.set_salt.assert_called_once_with("priorsalt")

    def test_set_session_clears_salt_when_no_sidecar(self, tmp_path: Path) -> None:
        recorder = mock.MagicMock()
        state = backend_fastapi.GlobalState()
        state.acquisition = recorder

        with mock.patch.object(
            backend_fastapi, "get_global_state", return_value=state
        ), mock.patch.object(
            backend_fastapi, "RECORDING_BASE", str(tmp_path)
        ), mock.patch.object(
            backend_fastapi.datetime,
            "date",
            mock.MagicMock(today=mock.MagicMock(return_value=datetime.date(2026, 5, 21))),
        ):
            asyncio.run(backend_fastapi.set_session(subject_id="t999", db=mock.MagicMock()))

        # New session has no sidecar → recorder.set_salt("") clears whatever
        # may have been set during a prior session in the same process.
        recorder.set_salt.assert_called_once_with("")
