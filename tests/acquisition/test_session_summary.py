"""Tests for the /api/v1/health/session_summary endpoint.

Uses synthetic JSON sidecars in a tmp_path + the backend's TestClient so the
diagnose_session_issues integration is end-to-end, no hardware needed.
"""

from __future__ import annotations

import datetime
import json
import sys
import types
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


CAMERA_IDS = ["CAM_A", "CAM_B", "CAM_C"]
BASE_TIMESTAMP_NS = 1_000_000_000_000
FRAME_PERIOD_NS = 34_400_000  # ~29 fps


def _make_trial_data(n_frames: int = 30) -> dict:
    """Build a single-trial JSON sidecar matching the schema json_parser expects."""
    n_cams = len(CAMERA_IDS)
    timestamps = []
    frame_ids = []
    for i in range(n_frames):
        timestamps.append(
            [BASE_TIMESTAMP_NS + i * FRAME_PERIOD_NS + c * 100 for c in range(n_cams)]
        )
        frame_ids.append([1000 + i for _ in range(n_cams)])
    return {
        "serials": list(CAMERA_IDS),
        "timestamps": timestamps,
        "frame_id": frame_ids,
        "real_times": [[0.0] * n_cams] * n_frames,
        "exposure_times": [15000] * n_cams,
        "frame_rates_requested": [30] * n_cams,
        "frame_rates_binning": [30] * n_cams,
        "camera_config_hash": "deadbeef",
        "camera_info": {c: {} for c in CAMERA_IDS},
        "meta_info": {},
        "system_info": {},
    }


@pytest.fixture
def session_with_trials(tmp_path: Path):
    """Create a session directory with three trial sidecars named {root}_{YYYYMMDD}_{HHMMSS}.json."""
    session_dir = tmp_path / "participant_1" / "20260422"
    session_dir.mkdir(parents=True)
    for i in range(3):
        data = _make_trial_data(n_frames=30)
        filename = f"trial_20260422_12000{i}.json"
        (session_dir / filename).write_text(json.dumps(data))
    return session_dir


@pytest.fixture
def configured_session(session_with_trials, monkeypatch: pytest.MonkeyPatch):
    state = backend_fastapi.get_global_state()

    class StubRecorder:
        def __init__(self):
            self.camera_config = {"camera-info": {}}
            self.cams = []

        def close(self):
            return None

        async def get_camera_status(self):
            return []

    stub = StubRecorder()
    monkeypatch.setattr(backend_fastapi, "FlirRecorder", lambda *a, **kw: stub)
    monkeypatch.setattr(
        backend_fastapi, "synchronize_to_datajoint", lambda *a, **kw: None
    )

    state.current_session = backend_fastapi.Session(
        participant_name="participant_1",
        session_date=datetime.date(2026, 4, 22),
        recording_path=str(session_with_trials),
    )
    state._health_cache = None
    state._health_cache_ts = 0.0

    return state


def test_session_summary_endpoint_returns_insights_list(configured_session) -> None:
    with TestClient(backend_fastapi.app) as client:
        r = client.get("/api/v1/health/session_summary")
    assert r.status_code == 200
    body = r.json()
    assert body["n_trials"] == 3
    assert "insights" in body
    assert "recommendations" in body
    assert "trial_findings" in body
    assert isinstance(body["insights"], list)
    assert isinstance(body["recommendations"], list)
    assert isinstance(body["trial_findings"], list)


def test_session_summary_returns_404_without_session(
    configured_session, monkeypatch
) -> None:
    state = configured_session
    monkeypatch.setattr(state, "current_session", None)
    with TestClient(backend_fastapi.app) as client:
        r = client.get("/api/v1/health/session_summary")
    assert r.status_code == 404


def test_session_summary_returns_empty_when_no_trials_yet(
    configured_session, tmp_path
) -> None:
    """A session that's been created but hasn't recorded yet returns an empty
    summary instead of crashing — this is the common case right after the
    operator sets a participant ID.
    """
    state = configured_session
    empty_dir = tmp_path / "participant_2" / "20260430"
    empty_dir.mkdir(parents=True)
    state.current_session = backend_fastapi.Session(
        participant_name="participant_2",
        session_date=datetime.date(2026, 4, 30),
        recording_path=str(empty_dir),
    )
    with TestClient(backend_fastapi.app) as client:
        r = client.get("/api/v1/health/session_summary")
    assert r.status_code == 200
    body = r.json()
    assert body["n_trials"] == 0
    assert body["insights"] == []
    assert body["recommendations"] == []
    assert body["trial_findings"] == []
