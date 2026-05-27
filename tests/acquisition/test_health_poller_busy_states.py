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
    "Resetting",
    "Reset complete. Waiting to reconfigure.",
]
_FREE_STATUSES = ["Idle", "Uninitialized", None, ""]


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

    def test_reset_cameras_does_not_read_attributes_off_self_cams(self) -> None:
        """After a failed configure (or a mid-trial disconnect), wrappers in
        ``self.cams`` may hold dead PySpin handles whose attribute reads
        raise ``CameraError``. ``reset_cameras`` is the operator's primary
        recovery path for exactly that situation, so it must not depend on
        those attributes being readable.
        """
        import PySpin
        from multi_camera.acquisition.flir_recording_api import FlirRecorder
        from simple_pyspin import CameraError

        class DeadCam:
            def __getattr__(self, name):
                raise CameraError(f"Camera property '{name}' is not readable")

        recorder = FlirRecorder.__new__(FlirRecorder)
        recorder.cams = [DeadCam(), DeadCam()]
        recorder.config_file = ""
        recorder.set_status = lambda s: None
        # Stub close so reset_cameras doesn't try to walk the dead cams there.
        recorder.close = lambda: None

        # Raise a marker from PySpin.System.GetInstance — if reset_cameras
        # reaches that call without crashing on self.cams attribute access
        # first, the marker propagates out, proving the fix.
        sentinel = RuntimeError("__got_past_self_cams_read__")
        fake_system_cls = mock.MagicMock()
        fake_system_cls.GetInstance.side_effect = sentinel
        with mock.patch.object(PySpin, "System", fake_system_cls, create=True):
            with pytest.raises(RuntimeError, match="__got_past_self_cams_read__"):
                asyncio.run(recorder.reset_cameras())

    def test_configure_cameras_restores_status_on_failure(self) -> None:
        """A failed configure must restore status to "Idle" before
        re-raising, otherwise every recovery endpoint stays 409'd because
        "Configuring" is in _BUSY_PYSPIN_STATES."""
        from multi_camera.acquisition.flir_recording_api import FlirRecorder

        seen: list[str] = []

        recorder = FlirRecorder.__new__(FlirRecorder)
        recorder.set_status = lambda status: seen.append(status)
        recorder.config_file = None
        recorder.excluded_serials = set()
        # Make the first PySpin call inside _configure_cameras_impl raise,
        # mimicking the wrong-subnet / dead-system failure mode.
        recorder.system = mock.MagicMock()
        recorder.system.GetInterfaces.side_effect = RuntimeError(
            "__pyspin_wedged__"
        )

        with pytest.raises(RuntimeError, match="__pyspin_wedged__"):
            asyncio.run(recorder.configure_cameras(num_cams=1))

        # Configuring was set first, then Idle must be restored before the
        # exception propagated. Without the fix, seen would be just
        # ["Configuring"] and the operator's recovery buttons would all
        # 409 forever.
        assert seen == ["Configuring", "Idle"], (
            "configure_cameras must restore status to Idle on failure; "
            f"got {seen!r}"
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


class TestPollerDeferredStart:
    """The HealthIdlePoller is constructed in lifespan but deliberately not
    started until update_config completes its first successful
    configure_cameras call. Before that, the poller's PySpin enumeration
    has nothing useful to compare against and adds latency / background
    churn on the GUI's mount-time fetches.
    """

    def test_update_config_with_a_yaml_starts_the_poller(self) -> None:
        recorder = mock.MagicMock()
        recorder.configure_cameras = mock.AsyncMock()
        poller = mock.MagicMock()

        state = backend_fastapi.GlobalState()
        state.acquisition = recorder
        state._health_poller = poller

        async def body():
            with mock.patch.object(
                backend_fastapi, "get_global_state", return_value=state
            ), mock.patch.object(
                backend_fastapi, "_refresh_health_after_configure", new=mock.AsyncMock()
            ):
                await backend_fastapi.update_config(
                    backend_fastapi.ConfigFileData(config="cotton_lab_config_12_cam.yaml")
                )

        asyncio.run(body())
        recorder.configure_cameras.assert_awaited_once()
        poller.start.assert_called_once()

    def test_update_config_empty_string_does_not_start_the_poller(self) -> None:
        """An empty config means 'reset' — no cameras to enumerate against,
        so the poller stays in whatever state it was in (not auto-started).
        """
        recorder = mock.MagicMock()
        recorder.reset = mock.MagicMock()
        poller = mock.MagicMock()

        state = backend_fastapi.GlobalState()
        state.acquisition = recorder
        state._health_poller = poller

        async def body():
            with mock.patch.object(
                backend_fastapi, "get_global_state", return_value=state
            ), mock.patch.object(
                backend_fastapi, "_refresh_health_after_configure", new=mock.AsyncMock()
            ):
                await backend_fastapi.update_config(
                    backend_fastapi.ConfigFileData(config="")
                )

        asyncio.run(body())
        recorder.reset.assert_called_once()
        poller.start.assert_not_called()
