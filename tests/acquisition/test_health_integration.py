"""Integration tests for the /api/v1/health/* endpoints.

Uses FastAPI's TestClient with a stub FlirRecorder so the full path through
run_health_check, the TTL cache, and the FastAPI response serialization is
exercised — no real cameras, no live DHCP, no Spinnaker SDK.

The Spinnaker SDK (PySpin / simple_pyspin) is a hardware-only C++ binding that
is typically absent in CI/dev environments. We stub it in ``sys.modules``
before importing the backend so the module graph loads.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest


def _install_pyspin_stubs() -> None:
    """Insert no-op PySpin / simple_pyspin modules into sys.modules if absent.

    The backend imports flir_recording_api, which imports PySpin + simple_pyspin
    at module scope. In environments without the Spinnaker SDK (CI, dev
    containers), we replace them with minimal stubs sufficient to let the
    module import.
    """
    for name in ("PySpin", "simple_pyspin"):
        if name in sys.modules:
            continue
        stub = types.ModuleType(name)
        if name == "simple_pyspin":

            class _Camera:  # noqa: D401 — stub
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("PySpin stub: Camera not available in tests")

            stub.Camera = _Camera  # type: ignore[attr-defined]
            stub._SYSTEM = None  # type: ignore[attr-defined]

            def _list_cameras():
                return []

            stub.list_cameras = _list_cameras  # type: ignore[attr-defined]
        sys.modules[name] = stub


_install_pyspin_stubs()


pytest.importorskip("fastapi.testclient")


def _import_backend_with_stubs():
    """Import the backend with /configs/ shimmed + a writable ./data/ dir.

    Production deploys these via bind mounts in Docker; in dev/CI we create
    what's needed so the module graph loads without patching the backend.
    """
    import os
    import unittest.mock as _mock

    real_listdir = os.listdir

    def safe_listdir(path):
        if str(path) == "/configs/" or str(path).startswith("/configs"):
            return []
        return real_listdir(path)

    # Repo-local data/ holds the SQLite recording DB. Test runs in CI create it
    # on demand; production uses a bind mount.
    os.makedirs("data", exist_ok=True)

    with _mock.patch("os.listdir", side_effect=safe_listdir):
        from fastapi.testclient import TestClient as _TestClient  # noqa: F401
        from multi_camera.acquisition.health import (
            DetectedCamera as _DetectedCamera,
            HealthConfig as _HealthConfig,
        )
        from multi_camera.backend import fastapi as _backend_fastapi
    return _TestClient, _DetectedCamera, _HealthConfig, _backend_fastapi


try:
    TestClient, DetectedCamera, HealthConfig, backend_fastapi = (
        _import_backend_with_stubs()
    )
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"Backend not importable in this environment: {exc}",
        allow_module_level=True,
    )


class StubRecorder:
    """A minimal stand-in for FlirRecorder used by the health endpoints."""

    def __init__(self, expected_serials: list[str] | None = None):
        expected_serials = expected_serials or []
        self.camera_config = {
            "camera-info": {s: {"lens_info": "stub"} for s in expected_serials}
        }
        self.cams: list[Any] = []

    def __call__(self, *args, **kwargs):
        # Compatibility: the real FlirRecorder is instantiated with a status
        # callback, so when this class is swapped in as the constructor it
        # ignores those args and returns self.
        return self

    async def get_camera_status(self):
        return []

    def close(self) -> None:
        return None


@pytest.fixture
def configured_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Wire a clean GlobalState + lease file + injected enumerator into the backend."""
    stub_recorder = StubRecorder(expected_serials=["111", "222"])

    # Replace FlirRecorder in the backend module so lifespan startup creates
    # our stub instead of a real PySpin-backed recorder.
    monkeypatch.setattr(
        backend_fastapi, "FlirRecorder", lambda *args, **kwargs: stub_recorder
    )
    monkeypatch.setattr(
        backend_fastapi, "synchronize_to_datajoint", lambda *a, **kw: None
    )

    state = backend_fastapi.get_global_state()
    state.recording_status = "Idle"
    state._health_cache = None
    state._health_cache_ts = 0.0

    lease_file = tmp_path / "dhcpd.leases"
    lease_file.write_text("lease 192.168.1.50 {\n  ends 1 2099/01/01 12:00:00;\n}\n")

    state.health_config = HealthConfig(
        deployment_mode="laptop",
        network_interface="enp5s0",
        dhcp_lease_file=lease_file,
        cache_ttl_s=0.5,
    )

    # The lifespan startup overwrites health_config from env; re-install after the
    # backend boots. Simpler: patch HealthConfig.from_env to return our config.
    monkeypatch.setattr(
        backend_fastapi.HealthConfig,
        "from_env",
        classmethod(lambda cls, env=None: state.health_config),
    )

    # Inject fake DHCP + camera data without touching subprocesses/PySpin.
    original_run = backend_fastapi.run_health_check

    enumerator_result = [
        DetectedCamera(serial="111", ip="192.168.1.51", link_speed_mbps=1000),
        DetectedCamera(serial="222", ip="192.168.1.52", link_speed_mbps=1000),
    ]

    def fake_run_health_check(*args, **kwargs):
        kwargs.setdefault(
            "ip_addr_runner", lambda iface, timeout_s=1.0: "    inet 192.168.1.1/24"
        )
        kwargs.setdefault("camera_enumerator", lambda: list(enumerator_result))
        return original_run(*args, **kwargs)

    monkeypatch.setattr(backend_fastapi, "run_health_check", fake_run_health_check)

    # check_host_network calls subprocess directly (no injection points); patch
    # the private helpers so the dev container's actual host config doesn't
    # show up as findings during tests.
    import multi_camera.acquisition.health as _health

    monkeypatch.setattr(
        _health,
        "_read_link_state",
        lambda iface, timeout_s=1.0: {"present": True, "carrier": True, "mtu": 9000},
    )
    monkeypatch.setattr(
        _health,
        "_read_sysctl_int",
        lambda key, timeout_s=1.0: 10_000_000 if "rmem" in key else None,
    )

    yield state, enumerator_result


def test_get_health_returns_ok_when_all_green(configured_backend) -> None:
    state, _ = configured_backend
    with TestClient(backend_fastapi.app) as client:
        r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["overall"] == "ok"
    assert body["dhcp"]["service_active"] is True
    assert body["cameras"]["missing"] == []
    assert body["recording_state"] == "Idle"


def test_get_health_dhcp_narrow(configured_backend) -> None:
    with TestClient(backend_fastapi.app) as client:
        r = client.get("/api/v1/health/dhcp")
    assert r.status_code == 200
    body = r.json()
    assert body["applicable"] is True
    assert body["service_active"] is True


def test_get_health_cameras_narrow(configured_backend) -> None:
    with TestClient(backend_fastapi.app) as client:
        r = client.get("/api/v1/health/cameras")
    assert r.status_code == 200
    body = r.json()
    assert body["missing"] == []
    assert set(body["detected"]) == {"111", "222"}


def test_cache_returns_identical_report_within_ttl(configured_backend) -> None:
    with TestClient(backend_fastapi.app) as client:
        r1 = client.get("/api/v1/health").json()
        r2 = client.get("/api/v1/health").json()
    assert r1["generated_at"] == r2["generated_at"]


def test_refresh_bypasses_cache(configured_backend) -> None:
    state, enumerator_result = configured_backend
    with TestClient(backend_fastapi.app) as client:
        r1 = client.get("/api/v1/health").json()
        # Simulate camera disappearing; POST /health/refresh should see the change.
        enumerator_result.pop()
        r2 = client.post("/api/v1/health/refresh").json()
    assert r1["cameras"]["missing"] == []
    assert r2["cameras"]["missing"] == ["222"]
    assert r2["overall"] == "warn"


def test_cache_expires_after_ttl(configured_backend) -> None:
    state, _ = configured_backend
    with TestClient(backend_fastapi.app) as client:
        r1 = client.get("/api/v1/health").json()
        time.sleep(state.health_config.cache_ttl_s + 0.1)
        r2 = client.get("/api/v1/health").json()
    # generated_at identical would mean TTL didn't expire.
    assert r1["generated_at"] != r2["generated_at"]
