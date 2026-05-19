"""Tests for the busy-state predicate that gates HealthIdlePoller's PySpin
GigE enumeration and the recovery endpoints (restart / restore / exclude),
and for the ``Configuring`` status flip inside ``FlirRecorder.configure_cameras``
that lets the predicate fire before ``init_camera``'s register writes."""

from __future__ import annotations

import asyncio
import sys
import types
import unittest.mock as mock

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


_BUSY_STATUSES = [
    "Configuring",
    "Synchronizing",
    "Synchronized",
    "Starting",
    "Recording",
]
_FREE_STATUSES = ["Idle", "Uninitialized", "Resetting", None, ""]


class TestShouldSkipCameraEnumeration:
    """The poller must skip PySpin GigE enumeration whenever the recorder is
    in a state that holds camera handles or runs PySpin register writes —
    otherwise enumeration races ``init_camera``'s ``LineMode`` /
    ``DeviceLinkThroughputLimit`` writes and raises ``GenICam::AccessException``."""

    @pytest.mark.parametrize("status", _BUSY_STATUSES)
    def test_skip_during_busy(self, status) -> None:
        assert backend_fastapi._should_skip_camera_enumeration(status) is True

    @pytest.mark.parametrize("status", _FREE_STATUSES)
    def test_do_not_skip_when_free(self, status) -> None:
        assert backend_fastapi._should_skip_camera_enumeration(status) is False


class TestFlirRecorderConfigureCameras:
    """``FlirRecorder.configure_cameras`` is the single locus for the
    Configuring flip: any future caller (HTTP handler, script, test) gets the
    correct status transition for free."""

    def test_configure_cameras_flips_to_configuring_before_pyspin_work(self) -> None:
        from multi_camera.acquisition.flir_recording_api import FlirRecorder

        seen: list[str] = []

        def flip_then_abort(status):
            seen.append(status)
            raise RuntimeError("__aborted_by_test__")

        recorder = FlirRecorder.__new__(FlirRecorder)
        recorder.set_status = flip_then_abort
        recorder.config_file = None
        recorder.excluded_serials = set()
        recorder.system = mock.MagicMock()

        with pytest.raises(RuntimeError, match="__aborted_by_test__"):
            asyncio.run(recorder.configure_cameras(num_cams=1))

        assert seen == ["Configuring"], (
            "configure_cameras' first observable side effect must be "
            f"set_status('Configuring'); got {seen!r}"
        )


class TestOperatorActionGuards:
    """Recovery endpoints (restart / restore-defaults / change-exclusion) must
    refuse with 409 during any PySpin-busy state, not just ``Recording`` —
    clicking 'Restart acquisition' 200ms after a config-change POST should not
    race configure_cameras."""

    def _state_with(self, status: str):
        state = backend_fastapi.GlobalState()
        state.recording_status = status
        state.acquisition = mock.MagicMock()
        return state

    @pytest.mark.parametrize("status", _BUSY_STATUSES)
    def test_restart_acquisition_409(self, status) -> None:
        from fastapi import HTTPException

        state = self._state_with(status)

        async def body():
            with mock.patch.object(
                backend_fastapi, "get_global_state", return_value=state
            ):
                with pytest.raises(HTTPException) as exc:
                    await backend_fastapi.restart_acquisition()
                assert exc.value.status_code == 409
                assert status in str(exc.value.detail)

        asyncio.run(body())

    @pytest.mark.parametrize("status", _BUSY_STATUSES)
    def test_restore_camera_defaults_409(self, status) -> None:
        from fastapi import HTTPException

        state = self._state_with(status)

        async def body():
            with mock.patch.object(
                backend_fastapi, "get_global_state", return_value=state
            ):
                with pytest.raises(HTTPException) as exc:
                    await backend_fastapi.restore_camera_defaults("12345")
                assert exc.value.status_code == 409
                assert status in str(exc.value.detail)

        asyncio.run(body())

    @pytest.mark.parametrize("status", _BUSY_STATUSES)
    def test_set_camera_excluded_409(self, status) -> None:
        from fastapi import HTTPException

        state = self._state_with(status)

        async def body():
            with mock.patch.object(
                backend_fastapi, "get_global_state", return_value=state
            ):
                with pytest.raises(HTTPException) as exc:
                    await backend_fastapi._set_camera_excluded("12345", True)
                assert exc.value.status_code == 409
                assert status in str(exc.value.detail)

        asyncio.run(body())
