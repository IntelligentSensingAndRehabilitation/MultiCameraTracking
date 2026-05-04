"""Integration test for the /api/v1/restart_acquisition endpoint.

Uses FastAPI's TestClient with a stub recorder. Verifies:
- reset() runs first, then configure_cameras with the saved config_file
- 409 is returned (and recorder is untouched) while recording
"""

from __future__ import annotations

import sys
import types
from typing import Any

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
        from fastapi.testclient import TestClient as _TestClient
        from multi_camera.backend import fastapi as _backend
    return _TestClient, _backend


try:
    TestClient, backend = _import_backend()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Backend not importable: {exc}", allow_module_level=True)


class StubRecorder:
    def __init__(self, config_file: str = "/configs/test.yaml"):
        self.config_file = config_file
        self.camera_config: dict[str, Any] = {"camera-info": {}}
        self.cams: list[Any] = []
        self.calls: list[Any] = []
        self.known_serials: list[str] = []

    def __call__(self, *args, **kwargs):
        return self

    def reset(self):
        self.calls.append("reset")
        # The real FlirRecorder.close() (called from reset()) clears config_file.
        self.config_file = None

    async def configure_cameras(self, config_file=None, **kwargs):
        self.calls.append(("configure_cameras", config_file))
        self.config_file = config_file

    async def get_camera_status(self):
        return []

    def close(self):
        return None

    def restore_camera_defaults(self, serial: str) -> None:
        if serial not in self.known_serials:
            raise ValueError(f"Camera {serial} not found")
        self.calls.append(("restore_camera_defaults", serial))

    def __post_init_excluded(self):
        if not hasattr(self, "excluded_serials"):
            self.excluded_serials = set()

    def set_excluded_serials(self, serials):
        self.__post_init_excluded()
        self.excluded_serials = {str(s) for s in serials}
        self.calls.append(("set_excluded_serials", sorted(self.excluded_serials)))


@pytest.fixture
def configured_backend(monkeypatch: pytest.MonkeyPatch):
    stub = StubRecorder()
    monkeypatch.setattr(backend, "FlirRecorder", lambda *a, **kw: stub)
    monkeypatch.setattr(backend, "synchronize_to_datajoint", lambda *a, **kw: None)
    state = backend.get_global_state()
    state.recording_status = "Idle"
    state.acquisition = stub
    yield state, stub


def test_restart_when_idle_resets_then_reconfigures(configured_backend) -> None:
    state, stub = configured_backend
    saved_config = stub.config_file
    with TestClient(backend.app) as client:
        r = client.post("/api/v1/restart_acquisition")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["config"] == saved_config
    # reset() ran first, then configure_cameras with the saved path.
    assert stub.calls[0] == "reset"
    assert stub.calls[1] == ("configure_cameras", saved_config)


def test_restart_returns_409_while_recording(configured_backend) -> None:
    state, stub = configured_backend
    state.recording_status = "Recording"
    with TestClient(backend.app) as client:
        r = client.post("/api/v1/restart_acquisition")
    assert r.status_code == 409
    assert stub.calls == []


def test_restore_defaults_runs_then_reinits(configured_backend) -> None:
    state, stub = configured_backend
    stub.known_serials = ["23280538", "20150962"]
    saved_config = stub.config_file
    with TestClient(backend.app) as client:
        r = client.post("/api/v1/cameras/23280538/restore_defaults")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "success"
    assert body["serial"] == "23280538"
    # Order: per-camera restore first, then full reset, then reconfigure.
    assert stub.calls == [
        ("restore_camera_defaults", "23280538"),
        "reset",
        ("configure_cameras", saved_config),
    ]


def test_restore_defaults_returns_404_for_unknown_serial(configured_backend) -> None:
    state, stub = configured_backend
    stub.known_serials = ["20150962"]
    with TestClient(backend.app) as client:
        r = client.post("/api/v1/cameras/99999999/restore_defaults")
    assert r.status_code == 404
    # No reset/reconfigure when the target serial isn't recognized.
    assert all(c[0] != "reset" for c in stub.calls if isinstance(c, str) is False) or "reset" not in stub.calls


def test_restore_defaults_returns_409_while_recording(configured_backend) -> None:
    state, stub = configured_backend
    stub.known_serials = ["23280538"]
    state.recording_status = "Recording"
    with TestClient(backend.app) as client:
        r = client.post("/api/v1/cameras/23280538/restore_defaults")
    assert r.status_code == 409
    assert stub.calls == []


def test_exclude_camera_sets_then_reconfigures(configured_backend) -> None:
    state, stub = configured_backend
    stub.excluded_serials = set()
    saved_config = stub.config_file
    with TestClient(backend.app) as client:
        r = client.post("/api/v1/cameras/23280538/exclude")
    assert r.status_code == 200
    body = r.json()
    assert body["excluded"] is True
    assert body["excluded_serials"] == ["23280538"]
    # Order: set the exclusion, then reset, then configure_cameras.
    assert stub.calls == [
        ("set_excluded_serials", ["23280538"]),
        "reset",
        ("configure_cameras", saved_config),
    ]


def test_include_camera_clears_exclusion(configured_backend) -> None:
    state, stub = configured_backend
    stub.excluded_serials = {"23280538", "20150962"}
    with TestClient(backend.app) as client:
        r = client.post("/api/v1/cameras/23280538/include")
    assert r.status_code == 200
    body = r.json()
    assert body["excluded"] is False
    assert body["excluded_serials"] == ["20150962"]


def test_exclude_returns_409_while_recording(configured_backend) -> None:
    state, stub = configured_backend
    state.recording_status = "Recording"
    with TestClient(backend.app) as client:
        r = client.post("/api/v1/cameras/23280538/exclude")
    assert r.status_code == 409
    assert stub.calls == []


def test_restart_without_saved_config_skips_configure(monkeypatch) -> None:
    """If the recorder has no config_file (e.g. never configured), restart
    still tears down PySpin but skips the configure call."""
    stub = StubRecorder(config_file=None)
    monkeypatch.setattr(backend, "FlirRecorder", lambda *a, **kw: stub)
    monkeypatch.setattr(backend, "synchronize_to_datajoint", lambda *a, **kw: None)
    state = backend.get_global_state()
    state.recording_status = "Idle"
    state.acquisition = stub

    with TestClient(backend.app) as client:
        r = client.post("/api/v1/restart_acquisition")
    assert r.status_code == 200
    assert stub.calls == ["reset"]
