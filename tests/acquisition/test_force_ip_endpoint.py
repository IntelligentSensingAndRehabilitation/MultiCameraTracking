"""Tests for ``POST /api/v1/cameras/{mac}/force_ip``.

The endpoint is the GUI surface for the same operation
``scripts/acquisition/force_camera_ips.py`` exposes from the CLI. Both
gate to laptop deployment mode (the upstream DHCP problem in network
mode wants a deliberate operator fix, not a volatile ForceIP). The
endpoint also refuses while the system is in any PySpin-busy state to
avoid racing ``configure_cameras`` or an active recording.
"""

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

from multi_camera.acquisition.health import HealthConfig


def _make_state(deployment_mode: str = "laptop", recording_status: str = "Idle"):
    state = backend_fastapi.GlobalState()
    state.health_config = HealthConfig(deployment_mode=deployment_mode)
    state.recording_status = recording_status
    state.acquisition = mock.MagicMock()
    state.acquisition.force_camera_ip = mock.MagicMock(
        return_value={"mac": "aa:bb:cc:dd:ee:ff", "ip": "192.168.1.240"}
    )
    return state


def _call_endpoint(state, mac: str, body: dict | None = None):
    """Drive the endpoint synchronously, returning either the result dict
    or the raised HTTPException for assertion.
    """
    from fastapi import HTTPException

    data = backend_fastapi.ForceIpData(**(body or {}))

    async def body_coro():
        with mock.patch.object(
            backend_fastapi, "get_global_state", return_value=state
        ), mock.patch.object(
            backend_fastapi, "broadcast_event", new=mock.MagicMock()
        ), mock.patch.object(
            backend_fastapi,
            "_refresh_health_after_configure",
            new=mock.AsyncMock(),
        ):
            return await backend_fastapi.force_camera_ip(mac, data)

    try:
        return asyncio.run(body_coro()), None
    except HTTPException as e:
        return None, e


class TestDeploymentModeGate:
    """Network mode is deliberately unsupported today — the script and
    endpoint both refuse so the operator doesn't paper over an upstream
    DHCP issue with a volatile ForceIP that has to be re-run every
    power-cycle.
    """

    def test_endpoint_refuses_in_network_mode(self) -> None:
        state = _make_state(deployment_mode="network")
        result, exc = _call_endpoint(state, "aa:bb:cc:dd:ee:ff")
        assert result is None
        assert exc is not None
        assert exc.status_code == 409
        assert "laptop" in exc.detail.lower()
        assert "network" in exc.detail.lower()
        # Must not have tried the force — gating is upstream of the call
        state.acquisition.force_camera_ip.assert_not_called()

    def test_endpoint_allows_in_laptop_mode(self) -> None:
        state = _make_state(deployment_mode="laptop")
        result, exc = _call_endpoint(state, "aa:bb:cc:dd:ee:ff")
        assert exc is None
        assert result == {
            "status": "success",
            "mac": "aa:bb:cc:dd:ee:ff",
            "ip": "192.168.1.240",
        }
        state.acquisition.force_camera_ip.assert_called_once()

    def test_endpoint_refuses_when_health_config_missing(self) -> None:
        """If lifespan hasn't populated health_config the gate becomes
        a no-op (default HealthConfig() has deployment_mode='laptop').
        Verify that path still works rather than NameError-ing.
        """
        state = backend_fastapi.GlobalState()
        state.recording_status = "Idle"
        state.acquisition = mock.MagicMock()
        state.acquisition.force_camera_ip = mock.MagicMock(
            return_value={"mac": "aa", "ip": "192.168.1.240"}
        )
        # state.health_config = default HealthConfig() — laptop mode
        result, exc = _call_endpoint(state, "aa")
        assert exc is None
        assert result["status"] == "success"


class TestBusyStateGate:
    """force_camera_ip touches PySpin and must not run while any other
    PySpin-busy operation is in flight.
    """

    @pytest.mark.parametrize(
        "status",
        ["Configuring", "Synchronizing", "Synchronized", "Starting", "Recording"],
    )
    def test_endpoint_refuses_in_pyspin_busy_state(self, status: str) -> None:
        state = _make_state(recording_status=status)
        result, exc = _call_endpoint(state, "aa:bb:cc:dd:ee:ff")
        assert result is None
        assert exc is not None
        assert exc.status_code == 409
        assert status in exc.detail
        state.acquisition.force_camera_ip.assert_not_called()

    @pytest.mark.parametrize("status", ["Idle", "Uninitialized"])
    def test_endpoint_proceeds_in_idle_or_uninitialized(self, status: str) -> None:
        state = _make_state(recording_status=status)
        result, exc = _call_endpoint(state, "aa:bb:cc:dd:ee:ff")
        assert exc is None
        assert result["status"] == "success"


class TestForceCameraIpInvocation:
    """The endpoint passes optional body fields through to
    ``state.acquisition.force_camera_ip`` and converts exceptions into
    the right HTTP status.
    """

    def test_no_body_passes_only_mac(self) -> None:
        state = _make_state()
        _, exc = _call_endpoint(state, "aa:bb:cc:dd:ee:ff", body={})
        assert exc is None
        kwargs = state.acquisition.force_camera_ip.call_args.kwargs
        assert kwargs == {"mac": "aa:bb:cc:dd:ee:ff"}

    def test_full_body_passes_all_fields(self) -> None:
        state = _make_state()
        _, exc = _call_endpoint(
            state,
            "aa:bb:cc:dd:ee:ff",
            body={
                "ip": "192.168.1.250",
                "mask": "255.255.255.0",
                "gateway": "192.168.1.1",
            },
        )
        assert exc is None
        kwargs = state.acquisition.force_camera_ip.call_args.kwargs
        assert kwargs == {
            "mac": "aa:bb:cc:dd:ee:ff",
            "ip": "192.168.1.250",
            "mask": "255.255.255.0",
            "gateway": "192.168.1.1",
        }

    def test_value_error_becomes_404(self) -> None:
        """force_camera_ip raises ValueError when no camera with the MAC
        is currently off-subnet — surface as 404 so the operator can
        retry after a discovery refresh.
        """
        state = _make_state()
        state.acquisition.force_camera_ip.side_effect = ValueError(
            "No off-subnet camera with MAC aa:bb:cc:dd:ee:ff on any interface"
        )
        _, exc = _call_endpoint(state, "aa:bb:cc:dd:ee:ff")
        assert exc is not None
        assert exc.status_code == 404

    def test_other_exception_becomes_500(self) -> None:
        state = _make_state()
        state.acquisition.force_camera_ip.side_effect = RuntimeError(
            "Camera ignored ForceIP, still off-subnet after 5s"
        )
        _, exc = _call_endpoint(state, "aa:bb:cc:dd:ee:ff")
        assert exc is not None
        assert exc.status_code == 500
        assert "Camera ignored" in exc.detail
