#!/usr/bin/env python3
"""
Database query tool for backup system - runs inside Docker

This script ONLY queries the database. All filesystem operations
are handled by the shell script on the host.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_camera.backend.recording_db import get_db, Participant, Session as SessionModel, Imported

# Import DataJoint schemas to check actual data (not just the Imported flag)
import datajoint as dj
from multi_camera.datajoint.sessions import Subject, Session as DJSession, Recording
from multi_camera.datajoint.multi_camera_dj import MultiCameraRecording, SingleCameraVideo


def check_datajoint_imported(participant_id: str, session_date: str) -> bool:
    """
    Check if session has been imported to DataJoint

    This checks BOTH:
    1. The SQLite Imported flag (quick check)
    2. The actual DataJoint tables contain the data (thorough check)

    Returns True only if data exists in both places.
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

    session = db.query(SessionModel).filter(
        SessionModel.participant_id == participant.id,
        SessionModel.session_date == session_date_obj
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
        print(f"Error checking DataJoint tables: {e}", file=sys.stderr)
        return False


def get_sessions_in_range(start_date: str = None, end_date: str = None):
    """Get all sessions in date range"""
    db = get_db()

    query = db.query(SessionModel)

    if start_date:
        start_date_obj = datetime.strptime(start_date, "%Y%m%d").date()
        query = query.filter(SessionModel.session_date >= start_date_obj)

    if end_date:
        end_date_obj = datetime.strptime(end_date, "%Y%m%d").date()
        query = query.filter(SessionModel.session_date <= end_date_obj)

    sessions = query.order_by(SessionModel.session_date.desc()).all()

    results = []
    for session in sessions:
        participant = db.query(Participant).filter(
            Participant.id == session.participant_id
        ).first()
        if participant:
            results.append({
                'participant_id': participant.name,
                'session_date': session.session_date.strftime("%Y%m%d"),
                'datajoint_imported': check_datajoint_imported(participant.name, session.session_date.strftime("%Y%m%d"))
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Database query tool for backup system")
    parser.add_argument('--check-datajoint', nargs=2, metavar=('PARTICIPANT', 'DATE'),
                        help='Check if session imported to DataJoint')
    parser.add_argument('--get-sessions', nargs='*', metavar='DATE',
                        help='Get sessions in range: --get-sessions [START] [END]')

    args = parser.parse_args()

    if args.check_datajoint:
        participant_id, session_date = args.check_datajoint
        imported = check_datajoint_imported(participant_id, session_date)
        print("1" if imported else "0")
        sys.exit(0 if imported else 1)

    elif args.get_sessions is not None:
        start_date = args.get_sessions[0] if len(args.get_sessions) > 0 else None
        end_date = args.get_sessions[1] if len(args.get_sessions) > 1 else None
        sessions = get_sessions_in_range(start_date, end_date)

        # Output as tab-separated values for shell script parsing
        for session in sessions:
            print(f"{session['participant_id']}\t{session['session_date']}\t{session['datajoint_imported']}")
        sys.exit(0)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
