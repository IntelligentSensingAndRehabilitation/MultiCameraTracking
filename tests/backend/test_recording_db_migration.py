"""Tests for the #28/#25 startup migration (_ensure_recording_metadata_columns).

Builds a legacy-schema SQLite DB with raw DDL (no recording_metadata/duration
columns) and verifies the ALTER + backfill, including the parse-&-strip of
app-written "10MWT: <n>s" comment text, and idempotency on re-run.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine, inspect, text

from multi_camera.backend.recording_db import (
    _ensure_recording_metadata_columns,
    _wrap_legacy_comment,
)

LEGACY_DDL = """
CREATE TABLE recordings (
    id INTEGER NOT NULL PRIMARY KEY,
    session_id INTEGER,
    filename VARCHAR,
    recording_timestamp DATETIME,
    comment VARCHAR,
    config_file VARCHAR,
    should_process BOOLEAN,
    timestamp_spread INTEGER
)
"""


@pytest.fixture
def legacy_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/legacy.db")
    with engine.begin() as conn:
        conn.execute(text(LEGACY_DDL))
        for row_id, comment in [
            (1, "baseline 10MWT: 12.34s"),
            (2, None),
            (3, "plain walking note"),
            (4, "10MWT: 9.87s 10MWT: 12.34s"),
            (5, "fast walk 10mwt: 5.50s"),  # case-insensitive
        ]:
            conn.execute(
                text("INSERT INTO recordings (id, comment) VALUES (:id, :comment)"),
                {"id": row_id, "comment": comment},
            )
    return engine


def _containers(engine):
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, recording_metadata FROM recordings ORDER BY id")
        ).fetchall()
    return {row_id: json.loads(meta) for row_id, meta in rows}


def test_migration_adds_columns_and_backfills(legacy_engine):
    _ensure_recording_metadata_columns(legacy_engine)

    cols = [c["name"] for c in inspect(legacy_engine).get_columns("recordings")]
    assert "recording_metadata" in cols
    assert "duration" in cols

    containers = _containers(legacy_engine)
    assert containers[1] == {"comment": "baseline", "10mwt_time": 12.34}
    assert containers[2] == {"comment": ""}
    assert containers[3] == {"comment": "plain walking note"}
    # Multiple matches: last one wins, all stripped
    assert containers[4] == {"comment": "", "10mwt_time": 12.34}
    assert containers[5] == {"comment": "fast walk", "10mwt_time": 5.5}

    with legacy_engine.begin() as conn:
        durations = conn.execute(text("SELECT DISTINCT duration FROM recordings")).fetchall()
    assert durations == [(None,)]  # historical rows stay NULL (no backfill)


def test_migration_is_idempotent(legacy_engine):
    _ensure_recording_metadata_columns(legacy_engine)
    before = _containers(legacy_engine)

    _ensure_recording_metadata_columns(legacy_engine)  # second run: early-returns
    assert _containers(legacy_engine) == before  # no double-wrap


def test_migration_noop_without_recordings_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/empty.db")
    _ensure_recording_metadata_columns(engine)  # must not raise
    assert "recordings" not in inspect(engine).get_table_names()


@pytest.mark.parametrize(
    "comment,expected",
    [
        (None, {"comment": ""}),
        ("", {"comment": ""}),
        ("note", {"comment": "note"}),
        ("10MWT: 12.34s", {"comment": "", "10mwt_time": 12.34}),
        ("a 10MWT: 12.34s b", {"comment": "a b", "10mwt_time": 12.34}),
        # Malformed number: leave the text untouched rather than guessing
        ("10MWT: 1.2.3s", {"comment": "10MWT: 1.2.3s"}),
    ],
)
def test_wrap_legacy_comment(comment, expected):
    assert _wrap_legacy_comment(comment) == expected
