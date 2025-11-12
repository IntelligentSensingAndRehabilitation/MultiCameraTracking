"""
Data integrity validation for MultiCameraTracking

Validates that session data exists in both:
1. SQLite database (local recording tracking)
2. DataJoint database (centralized storage)

This module provides both:
- Validation functions that can be imported by other tools
- Pytest tests for automated validation

Usage as tests:
    pytest tests/test_data_integrity.py
    pytest tests/test_data_integrity.py::test_all_sessions_imported -v
    pytest tests/test_data_integrity.py::test_recent_sessions_imported -v

Usage as library:
    from tests.test_data_integrity import check_session_in_datajoint
    result = check_session_in_datajoint("p001", "20250104")

Usage from CLI wrapper:
    python scripts/data_integrity_cli.py check-datajoint p001 20250104
    python scripts/data_integrity_cli.py get-sessions 20250101 20250131
"""

import pytest
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any

from multi_camera.backend.recording_db import (
    get_db,
    Participant,
    Session as SQLiteSession,
    Imported
)
from multi_camera.datajoint.sessions import Subject, Session as DJSession, Recording
from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, SingleCameraVideo


# ============================================================================
# Core Validation Functions (can be imported by other tools)
# ============================================================================

def check_session_in_datajoint(participant_id: str, session_date: str) -> bool:
    """
    Check if session has been imported to DataJoint

    This validates data integrity by checking BOTH:
    1. The SQLite Imported flag (quick check)
    2. The actual DataJoint tables contain the data (thorough check)

    Returns True only if data exists in both places.

    Args:
        participant_id: Participant identifier (e.g., 'p001')
        session_date: Session date in YYYYMMDD format (e.g., '20250104')

    Returns:
        True if session is fully imported to both SQLite and DataJoint, False otherwise

    Example:
        >>> check_session_in_datajoint("p001", "20250104")
        True
    """
    # Parse session date
    if isinstance(session_date, str):
        session_date_obj = datetime.strptime(session_date, "%Y%m%d").date()
    else:
        session_date_obj = session_date

    # First check the SQLite database (Recording database tracking)
    db = get_db()
    participant = db.query(Participant).filter(Participant.name == participant_id).first()
    if not participant:
        return False

    session = db.query(SQLiteSession).filter(
        SQLiteSession.participant_id == participant.id,
        SQLiteSession.session_date == session_date_obj
    ).first()

    if not session:
        return False

    imported = db.query(Imported).filter(Imported.session_id == session.id).first()
    if not imported:
        return False

    # Now check the actual DataJoint tables contain the data
    # This is the source of truth for the actual recording data
    try:
        # Check if Subject exists in DataJoint
        subject_key = {"participant_id": participant_id}
        if not (Subject & subject_key):
            return False

        # Check if Session exists in DataJoint
        session_key = {"participant_id": participant_id, "session_date": session_date_obj}
        if not (DJSession & session_key):
            return False

        # Check if Recording entries exist (this links Session to MultiCameraRecording)
        recordings = Recording & session_key
        if not recordings:
            return False

        # Check that the MultiCameraRecording entries exist
        for recording_key in recordings.fetch("KEY"):
            mcr_key = {
                "recording_timestamps": recording_key["recording_timestamps"],
                "camera_config_hash": recording_key["camera_config_hash"]
            }
            if not (MultiCameraRecording & mcr_key):
                return False

            # Check that SingleCameraVideo entries exist for this recording
            if not (SingleCameraVideo & mcr_key):
                return False

        # All checks passed - data exists in DataJoint
        return True

    except Exception as e:
        # Any exception during DataJoint checks means data is not accessible
        import sys
        print(f"Error checking DataJoint tables: {e}", file=sys.stderr)
        return False


def get_sessions_in_range(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get all sessions in date range with DataJoint import status

    Args:
        start_date: Optional start date in YYYYMMDD format (inclusive)
        end_date: Optional end date in YYYYMMDD format (inclusive)

    Returns:
        List of dicts with keys: participant_id, session_date, datajoint_imported

    Example:
        >>> sessions = get_sessions_in_range("20250101", "20250131")
        >>> for s in sessions:
        ...     print(f"{s['participant_id']}/{s['session_date']}: {s['datajoint_imported']}")
    """
    db = get_db()

    query = db.query(SQLiteSession)

    if start_date:
        start_date_obj = datetime.strptime(start_date, "%Y%m%d").date()
        query = query.filter(SQLiteSession.session_date >= start_date_obj)

    if end_date:
        end_date_obj = datetime.strptime(end_date, "%Y%m%d").date()
        query = query.filter(SQLiteSession.session_date <= end_date_obj)

    sessions = query.order_by(SQLiteSession.session_date.desc()).all()

    results = []
    for session in sessions:
        participant = db.query(Participant).filter(
            Participant.id == session.participant_id
        ).first()
        if participant:
            results.append({
                'participant_id': participant.name,
                'session_date': session.session_date.strftime("%Y%m%d"),
                'datajoint_imported': check_session_in_datajoint(
                    participant.name,
                    session.session_date.strftime("%Y%m%d")
                )
            })

    return results


# ============================================================================
# Pytest Tests
# ============================================================================

def test_all_sessions_imported():
    """
    Validate that all sessions in the database are properly imported to DataJoint

    This test ensures data integrity across the entire recording database.
    Failures indicate sessions that exist in SQLite but are not fully in DataJoint.
    """
    all_sessions = get_sessions_in_range()
    failed_sessions = [s for s in all_sessions if not s['datajoint_imported']]

    assert len(failed_sessions) == 0, (
        f"Found {len(failed_sessions)} sessions not fully imported to DataJoint:\n" +
        "\n".join([f"  - {s['participant_id']}/{s['session_date']}" for s in failed_sessions])
    )


def test_recent_sessions_imported():
    """
    Validate that all sessions from the last 30 days are imported to DataJoint

    This test focuses on recent data to catch import pipeline issues quickly.
    Use this for regular CI/CD checks.
    """
    end_date = date.today().strftime("%Y%m%d")
    start_date = (date.today() - timedelta(days=30)).strftime("%Y%m%d")
    recent_sessions = get_sessions_in_range(start_date, end_date)
    failed_sessions = [s for s in recent_sessions if not s['datajoint_imported']]

    assert len(failed_sessions) == 0, (
        f"Found {len(failed_sessions)} recent sessions (last 30 days) not imported:\n" +
        "\n".join([f"  - {s['participant_id']}/{s['session_date']}" for s in failed_sessions])
    )


def test_datajoint_connection():
    """
    Verify that DataJoint connection is working

    This basic sanity check ensures the test environment can connect to DataJoint.
    """
    try:
        # Try to access a DataJoint table
        _ = Subject.fetch(limit=1)
        # If we get here, connection works
        assert True
    except Exception as e:
        pytest.fail(f"DataJoint connection failed: {e}")


def test_sqlite_connection():
    """
    Verify that SQLite database connection is working

    This basic sanity check ensures the test environment can access the recording database.
    """
    try:
        db = get_db()
        # Try a simple query
        _ = db.query(Participant).first()
        # If we get here, connection works
        assert True
    except Exception as e:
        pytest.fail(f"SQLite connection failed: {e}")


