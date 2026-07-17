"""Tests for /api/v1/prior_recordings and /api/v1/update_recording (#28/#25).

Greenfield coverage for the recordings API: metadata container shape on GET,
whole-container round-trip (including unknown keys) on update, 404 on missing
recordings, and /calibrate accepting the new body. Uses a tmp-SQLite session via
app.dependency_overrides — no hardware, no DataJoint.
"""

from __future__ import annotations

import datetime
import sys
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
        from fastapi.testclient import TestClient as _TestClient
        from multi_camera.backend import fastapi as _backend_fastapi
    return _TestClient, _backend_fastapi


try:
    TestClient, backend_fastapi = _import_backend()
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Backend not importable: {exc}", allow_module_level=True)


from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402

from multi_camera.backend.recording_db import Base, Recording, add_recording  # noqa: E402

SESSION_DATE = datetime.date(2026, 7, 16)
TIMESTAMP = datetime.datetime(2026, 7, 16, 14, 30, 52)
FILENAME = "rec/P001_10mwt_20260716_143052"


@pytest.fixture
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/recordings.db")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()

    add_recording(
        session,
        participant_name="P001",
        session_date=SESSION_DATE,
        session_path="rec",
        filename=FILENAME,
        recording_timestamp=TIMESTAMP,
        config_file="config.yaml",
        comment="Baseline",
        timestamp_spread=0.003,
        duration=14.2,
    )
    yield session
    session.close()


@pytest.fixture
def client(db_session, monkeypatch):
    def override_db():
        yield db_session

    backend_fastapi.app.dependency_overrides[backend_fastapi.db_dependency] = override_db

    # Same stubbing as test_session_summary: the lifespan constructs a FlirRecorder
    # at startup and calls .close() at shutdown, which would touch the PySpin stub.
    class StubRecorder:
        def __init__(self):
            self.camera_config = {"camera-info": {}}
            self.cams = []

        def close(self):
            return None

        async def get_camera_status(self):
            return []

    monkeypatch.setattr(backend_fastapi, "FlirRecorder", lambda *a, **kw: StubRecorder())
    monkeypatch.setattr(backend_fastapi, "synchronize_to_datajoint", lambda *a, **kw: None)

    state = backend_fastapi.get_global_state()
    previous_session = state.current_session
    state.current_session = backend_fastapi.Session(
        participant_name="P001",
        session_date=SESSION_DATE,
        recording_path="rec",
    )

    with TestClient(backend_fastapi.app) as test_client:
        yield test_client

    state.current_session = previous_session
    backend_fastapi.app.dependency_overrides.pop(backend_fastapi.db_dependency, None)


def test_prior_recordings_returns_metadata_container_and_duration(client):
    r = client.get("/api/v1/prior_recordings")
    assert r.status_code == 200
    (entry,) = r.json()

    assert entry["filename"] == FILENAME
    assert entry["metadata"]["comment"] == "Baseline"
    assert entry["metadata"]["10mwt_time"] is None  # wire key is the alias
    assert entry["duration"] == 14.2
    assert "comment" not in entry  # flat field is gone from the wire shape


def test_update_recording_round_trips_whole_container(client, db_session):
    payload = {
        "participant": "P001",
        "filename": FILENAME,
        "recording_timestamp": TIMESTAMP.isoformat(),
        "metadata": {"comment": "Edited", "10mwt_time": 12.34, "custom": "kept"},
        "config_file": "config.yaml",
        "should_process": False,
        "timestamp_spread": 0.003,
    }
    r = client.post("/api/v1/update_recording", json=payload)
    assert r.status_code == 200

    db_session.expire_all()
    row = db_session.query(Recording).one()
    assert row.recording_metadata == {
        "comment": "Edited",
        "10mwt_time": 12.34,
        "custom": "kept",
    }
    assert row.should_process is False
    assert row.duration == 14.2  # server-computed; update must not touch it

    r = client.get("/api/v1/prior_recordings")
    (entry,) = r.json()
    assert entry["metadata"]["10mwt_time"] == 12.34
    assert entry["metadata"]["custom"] == "kept"


def test_update_recording_unknown_filename_returns_404(client):
    payload = {
        "participant": "P001",
        "filename": "rec/does_not_exist",
        "recording_timestamp": TIMESTAMP.isoformat(),
        "metadata": {"comment": "x"},
        "config_file": "",
        "should_process": True,
        "timestamp_spread": 0.0,
    }
    r = client.post("/api/v1/update_recording", json=payload)
    assert r.status_code == 404


def test_calibrate_accepts_container_body(client, monkeypatch):
    calibrate_stub = types.ModuleType("multi_camera.datajoint.calibrate_cameras")
    calibrate_stub.run_calibration = lambda **kwargs: None
    monkeypatch.setitem(
        sys.modules, "multi_camera.datajoint.calibrate_cameras", calibrate_stub
    )

    payload = {
        "participant": "P001",
        "filename": FILENAME,
        "recording_timestamp": TIMESTAMP.isoformat(),
        "metadata": {"comment": "charuco"},
        "config_file": "config.yaml",
        "should_process": True,
        "timestamp_spread": 0.003,
    }
    r = client.post("/api/v1/calibrate", params={"charuco_flag": True}, json=payload)
    assert r.status_code == 200
