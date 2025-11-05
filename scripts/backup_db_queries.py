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


def check_datajoint_imported(participant_id: str, session_date: str) -> bool:
    """Check if session has been imported to DataJoint"""
    db = get_db()

    participant = db.query(Participant).filter(Participant.name == participant_id).first()
    if not participant:
        return False

    if isinstance(session_date, str):
        session_date_obj = datetime.strptime(session_date, "%Y%m%d").date()
    else:
        session_date_obj = session_date

    session = db.query(SessionModel).filter(
        SessionModel.participant_id == participant.id,
        SessionModel.session_date == session_date_obj
    ).first()

    if not session:
        return False

    imported = db.query(Imported).filter(Imported.session_id == session.id).first()
    return imported is not None


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
