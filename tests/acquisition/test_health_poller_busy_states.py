"""Tests for the health-poller busy-state predicate that skips PySpin GigE
enumeration during configure / sync / recording, and for the ``Configuring``
status flip that lets the predicate fire during ``configure_cameras``."""

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


class TestShouldSkipCameraEnumeration:
    """The poller must skip PySpin GigE enumeration whenever the recorder is
    in a state that holds camera handles or runs PySpin writes — otherwise
    enumeration races ``init_camera``'s ``LineMode``/``DeviceLinkThroughputLimit``
    writes and raises ``GenICam::AccessException``."""

    @pytest.mark.parametrize(
        "status,expected",
        [
            ("Configuring", True),
            ("Synchronizing", True),
            ("Synchronized", True),
            ("Starting", True),
            ("Recording", True),
            ("Idle", False),
            ("Uninitialized", False),
            ("Resetting", False),
            (None, False),
            ("", False),
        ],
    )
    def test_skip_decision_per_status(self, status, expected) -> None:
        assert backend_fastapi._should_skip_camera_enumeration(status) is expected

    def test_pyspin_states_is_superset_of_recording_states(self) -> None:
        assert backend_fastapi._BUSY_RECORDING_STATES.issubset(
            backend_fastapi._BUSY_PYSPIN_STATES
        )


class TestConfigureCamerasFlipsToConfiguring:
    """Both ``configure_cameras`` call sites must flip status to ``Configuring``
    before awaiting the configure, so the poller skip predicate evaluates to
    True during ``init_camera``'s PySpin enumeration."""

    def test_update_config_sets_configuring_before_configure(self) -> None:
        from fastapi import HTTPException  # noqa: F401  (matches endpoint impl)

        recorder = mock.MagicMock()
        recorder.config_file = None
        call_order = []
        recorder.set_status.side_effect = lambda s: call_order.append(("set_status", s))

        async def fake_configure(*args, **kwargs):
            call_order.append(("configure_cameras", args, kwargs))

        recorder.configure_cameras = fake_configure

        state = backend_fastapi.GlobalState()
        state.acquisition = recorder

        async def body():
            with mock.patch.object(
                backend_fastapi, "get_global_state", return_value=state
            ), mock.patch.object(
                backend_fastapi,
                "_refresh_health_after_configure",
                new=mock.AsyncMock(),
            ):
                cfg = backend_fastapi.ConfigFileData(config="cotton_lab.yaml")
                await backend_fastapi.update_config(cfg)

        asyncio.run(body())

        statuses = [entry[1] for entry in call_order if entry[0] == "set_status"]
        configure_idx = next(
            i for i, entry in enumerate(call_order) if entry[0] == "configure_cameras"
        )
        set_status_idx = next(
            i for i, entry in enumerate(call_order) if entry[0] == "set_status"
        )
        assert "Configuring" in statuses
        assert set_status_idx < configure_idx, (
            "set_status('Configuring') must run BEFORE configure_cameras to "
            "cover the init_camera enumeration window."
        )

    def test_reset_and_configure_sets_configuring_before_configure(self) -> None:
        recorder = mock.MagicMock()
        call_order = []
        recorder.set_status.side_effect = lambda s: call_order.append(("set_status", s))
        recorder.reset.side_effect = lambda: call_order.append(("reset",))

        async def fake_configure(saved_config):
            call_order.append(("configure_cameras", saved_config))

        recorder.configure_cameras = fake_configure

        state = backend_fastapi.GlobalState()
        state.acquisition = recorder

        async def body():
            with mock.patch.object(
                backend_fastapi, "get_global_state", return_value=state
            ):
                await backend_fastapi._reset_and_configure(
                    saved_config="/configs/some.yaml", action="Restart"
                )

        asyncio.run(body())

        statuses = [entry[1] for entry in call_order if entry[0] == "set_status"]
        configure_idx = next(
            i for i, entry in enumerate(call_order) if entry[0] == "configure_cameras"
        )
        configuring_idx = next(
            i
            for i, entry in enumerate(call_order)
            if entry[0] == "set_status" and entry[1] == "Configuring"
        )
        assert "Configuring" in statuses
        assert configuring_idx < configure_idx, (
            "set_status('Configuring') must run BEFORE configure_cameras."
        )
