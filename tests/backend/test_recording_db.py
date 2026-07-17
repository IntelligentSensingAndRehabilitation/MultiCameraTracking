"""Unit tests for the recordings SQLite layer (#28 metadata container, #25 duration).

Pure sqlalchemy/pydantic — no camera hardware, no PySpin stubs, no DataJoint
connection (the push test stubs the lazily-imported sessions module).
"""

from __future__ import annotations

import datetime
import sys
import types

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from multi_camera.backend import recording_db
from multi_camera.backend.recording_db import (
    Base,
    Recording,
    RecordingMetadata,
    RecordingOut,
    ParticipantOut,
    _recording_to_out,
    add_recording,
    get_recordings,
    modify_recording_entry,
    push_to_datajoint,
)

SESSION_DATE = datetime.date(2026, 7, 16)
TIMESTAMP = datetime.datetime(2026, 7, 16, 14, 30, 52)


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/recordings.db")
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(autocommit=False, autoflush=False, bind=engine)()
    yield session
    session.close()


def _seed(db, comment="hello", filename="rec/trial_01", duration=14.2):
    return add_recording(
        db,
        participant_name="P001",
        session_date=SESSION_DATE,
        session_path="rec",
        filename=filename,
        recording_timestamp=TIMESTAMP,
        config_file="config.yaml",
        comment=comment,
        timestamp_spread=0.003,
        duration=duration,
    )


def test_add_recording_wraps_comment_and_stores_duration(db):
    _seed(db)

    row = db.query(Recording).one()
    assert row.recording_metadata == {"comment": "hello"}
    assert row.duration == 14.2
    assert row.comment is None  # legacy column no longer written


def test_get_recordings_exposes_metadata_and_duration(db):
    _seed(db)

    participants = get_recordings(db, participant_name="P001")
    recording = participants[0].sessions[0].recordings[0]
    assert recording.metadata.comment == "hello"
    assert recording.metadata.ten_mwt_time is None
    assert recording.duration == 14.2


def test_recording_to_out_falls_back_to_legacy_comment():
    # A row written by a pre-#28 binary after the column migration ran: the
    # container is NULL but the legacy comment column is set.
    row = Recording(
        filename="rec/legacy",
        recording_timestamp=TIMESTAMP,
        comment="legacy text",
        recording_metadata=None,
        config_file=None,
        should_process=True,
        timestamp_spread=None,
        duration=None,
    )
    out = _recording_to_out(row)
    assert out.metadata.comment == "legacy text"
    assert out.duration is None


def test_modify_recording_entry_replaces_container_and_preserves_unknown_keys(db):
    _seed(db)

    updated = RecordingOut(
        filename="rec/trial_01",
        recording_timestamp=TIMESTAMP,
        metadata=RecordingMetadata(
            comment="edited", **{"10mwt_time": 12.34, "future_key": 1}
        ),
        config_file="config.yaml",
        should_process=False,
        timestamp_spread=0.003,
    )
    modify_recording_entry(db, ParticipantOut(name="P001", sessions=[]), updated)

    row = db.query(Recording).one()
    assert row.recording_metadata == {
        "comment": "edited",
        "10mwt_time": 12.34,
        "future_key": 1,
    }
    assert row.should_process is False
    assert row.duration == 14.2  # server-computed field untouched by updates


def test_modify_recording_entry_raises_when_recording_missing(db):
    updated = RecordingOut(
        filename="rec/nope",
        recording_timestamp=TIMESTAMP,
        metadata=RecordingMetadata(comment="x"),
        config_file="",
        should_process=True,
        timestamp_spread=0.0,
    )
    with pytest.raises(ValueError, match="Recording not found"):
        modify_recording_entry(db, ParticipantOut(name="P001", sessions=[]), updated)


def test_push_to_datajoint_tuple_shapes(db, tmp_path, monkeypatch):
    _seed(db, comment="walk trial", filename="rec/trial_01")
    _seed(db, comment="charuco", filename="rec/calibration_20260716_140000", duration=None)

    captured = {}

    def fake_import_session(participant_id, session_date, video_project, recordings, fin, photo):
        captured["recordings"] = recordings

    sessions_stub = types.ModuleType("multi_camera.datajoint.sessions")
    sessions_stub.import_session = fake_import_session
    sessions_stub.PhotoSpec = types.SimpleNamespace
    monkeypatch.setitem(sys.modules, "multi_camera.datajoint.sessions", sessions_stub)

    calibration_calls = []
    monkeypatch.setattr(recording_db, "get_datajoint_external_path", lambda: str(tmp_path))
    monkeypatch.setattr(recording_db, "check_datajoint_external_mounted", lambda p: None)
    monkeypatch.setattr(recording_db, "synchronize_to_datajoint", lambda db: None)
    monkeypatch.setattr(
        recording_db,
        "_push_calibration_videos",
        lambda recs, trial_video_project: calibration_calls.append(recs),
    )

    push_to_datajoint(db, "P001", SESSION_DATE, video_project="test_project")

    # Trial tuples carry (filename, comment, whole container); calibration stays 2-tuples.
    assert captured["recordings"] == [
        ("rec/trial_01", "walk trial", {"comment": "walk trial", "10mwt_time": None})
    ]
    assert calibration_calls == [[("rec/calibration_20260716_140000", "charuco")]]
