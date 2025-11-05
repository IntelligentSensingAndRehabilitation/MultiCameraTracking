#!/usr/bin/env python3
"""
Backup sync tool for MultiCameraTracking sessions

Usage:
  backup_sync.py --start-date 20250101 [--end-date 20250131] [--dry-run]
  backup_sync.py --session <participant_id> <session_date> [--dry-run]
  backup_sync.py --verify <participant_id> <session_date>
  backup_sync.py --status <participant_id> <session_date>
  backup_sync.py --status <session_date>  # Show all sessions for date
  backup_sync.py --status-range  # Show all sessions
  backup_sync.py --status-range <start_date> [end_date]  # Show sessions in range
  backup_sync.py --safe-delete <participant_id> <session_date> [--force]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from multi_camera.backend.backup_manager import BackupManager
from multi_camera.backend.recording_db import get_db


def sync_single_session(args, manager):
    """Sync a single session"""
    participant_id, session_date = args.session
    print(f"Syncing {participant_id}/{session_date}...")

    result = manager.sync_session(participant_id, session_date, dry_run=args.dry_run)

    if result['success']:
        print(f"✓ Sync succeeded")
        print(f"  Files: {result['files_synced']}")
        print(f"  Size: {result['bytes_transferred']:,} bytes")
        print(f"  Duration: {result['duration_seconds']:.1f}s")

        if result.get('verification'):
            ver = result['verification']
            if ver['verified']:
                print(f"✓ Verification passed")
            else:
                print(f"✗ Verification failed")
                print(f"  File count match: {ver['file_count_match']}")
                print(f"  Size match: {ver['size_match']}")
                if ver['missing_files']:
                    print(f"  Missing files: {ver['missing_files']}")
    else:
        print(f"✗ Sync failed: {result['error']}")
        sys.exit(1)


def sync_multiple_sessions(args, manager):
    """Sync multiple sessions in date range"""
    print(f"Syncing sessions from {args.start_date}" + (f" to {args.end_date}" if args.end_date else ""))

    result = manager.sync_multiple_sessions(
        args.start_date,
        args.end_date,
        dry_run=args.dry_run
    )

    print(f"\nProcessed: {result['sessions_processed']}")
    print(f"Succeeded: {result['sessions_succeeded']}")
    print(f"Failed: {result['sessions_failed']}")

    if result['sessions_failed'] > 0:
        print("\nFailed sessions:")
        for session_result in result['results']:
            if not session_result['success']:
                print(f"  {session_result['participant_id']}/{session_result['session_date']}")
                print(f"    Error: {session_result['error']}")


def verify_backup(args, manager):
    """Verify backup for a session"""
    participant_id, session_date = args.verify
    print(f"Verifying backup for {participant_id}/{session_date}...")

    verification = manager.verify_backup(participant_id, session_date)

    if verification['verified']:
        print(f"✓ Backup verified")
    else:
        print(f"✗ Backup verification failed")

    print(f"  File count match: {verification['file_count_match']}")
    print(f"  Source files: {verification['source_file_count']}")
    print(f"  Dest files: {verification['dest_file_count']}")
    print(f"  Size match: {verification['size_match']}")
    print(f"  Total size: {verification['total_size']:,} bytes")

    if verification['missing_files']:
        print(f"  Missing files: {len(verification['missing_files'])}")
        for f in verification['missing_files'][:10]:
            print(f"    - {f}")


def show_status(args, manager):
    """Show comprehensive status for a session or all sessions in date range"""
    if len(args.status) == 1:
        date_arg = args.status[0]
        sessions = manager.get_sessions_in_range(start_date=date_arg, end_date=date_arg)

        if not sessions:
            print(f"No sessions found for date {date_arg}")
            return

        print(f"Status for all sessions on {date_arg}:\n")
        print(f"{'Date':<12} {'Participant':<15} {'DJ':<4} {'Backup':<8} {'Verified':<10} {'Safe':<6} {'Size (GB)':<10}")
        print("=" * 80)

        for participant_id, sess_date in sessions:
            dj_imported = manager.check_datajoint_imported(participant_id, sess_date)
            backup_exists = manager.backup_exists(participant_id, sess_date)

            if backup_exists:
                verification = manager.verify_backup(participant_id, sess_date)
                verified = verification['verified']
                size_gb = verification['total_size'] / 1e9
            else:
                verified = False
                size_gb = 0

            safe, _ = manager.is_safe_to_delete(participant_id, sess_date)

            dj_mark = '✓' if dj_imported else '✗'
            backup_mark = '✓' if backup_exists else '✗'
            verified_mark = '✓' if verified else '✗'
            safe_mark = '✓' if safe else '✗'

            print(f"{sess_date:<12} {participant_id:<15} {dj_mark:<4} {backup_mark:<8} {verified_mark:<10} {safe_mark:<6} {size_gb:>9.2f}")

    else:
        participant_id, session_date = args.status
        print(f"Status for {participant_id}/{session_date}:\n")

        dj_imported = manager.check_datajoint_imported(participant_id, session_date)
        print(f"DataJoint imported: {'✓' if dj_imported else '✗'}")

        backup_exists = manager.backup_exists(participant_id, session_date)
        print(f"Backup exists: {'✓' if backup_exists else '✗'}")

        if backup_exists:
            verification = manager.verify_backup(participant_id, session_date)
            print(f"Backup verified: {'✓' if verification['verified'] else '✗'}")
            print(f"  Files: {verification['source_file_count']}")
            print(f"  Size: {verification['total_size']:,} bytes")
            if not verification['verified']:
                print(f"  File count match: {verification['file_count_match']}")
                print(f"  Size match: {verification['size_match']}")

        safe, reason = manager.is_safe_to_delete(participant_id, session_date)
        print(f"\nSafe to delete: {'✓' if safe else '✗'}")
        if not safe:
            print(f"  Reason: {reason}")


def safe_delete_session(args, manager):
    """Delete session after safety checks"""
    participant_id, session_date = args.safe_delete

    safe, reason = manager.is_safe_to_delete(participant_id, session_date)

    if not safe and not args.force:
        print(f"✗ Session not safe to delete: {reason}")
        print("Use --force to delete anyway (not recommended)")
        sys.exit(1)

    source = manager.config.get_source_path(participant_id, session_date)

    if not source.exists():
        print(f"✗ Source path not found: {source}")
        sys.exit(1)

    files = list(source.rglob('*'))
    files = [f for f in files if f.is_file()]
    total_size = sum(f.stat().st_size for f in files)

    print(f"About to DELETE:")
    print(f"  Path: {source}")
    print(f"  Files: {len(files)}")
    print(f"  Size: {total_size / 1e9:.2f} GB")
    print(f"\nDataJoint imported: {'✓' if manager.check_datajoint_imported(participant_id, session_date) else '✗'}")

    backup_exists = manager.backup_exists(participant_id, session_date)
    if backup_exists:
        verification = manager.verify_backup(participant_id, session_date)
        print(f"Backup verified: {'✓' if verification['verified'] else '✗'}")
    else:
        print(f"Backup exists: ✗")

    response = input("\nType 'DELETE' to confirm: ")
    if response != 'DELETE':
        print("Cancelled")
        sys.exit(0)

    import shutil
    shutil.rmtree(source)
    print(f"✓ Deleted {source}")


def show_status_range(args, manager):
    """Show status for all sessions in a date range"""
    start_date = args.status_range[0] if args.status_range and len(args.status_range) > 0 else None
    end_date = args.status_range[1] if args.status_range and len(args.status_range) > 1 else None

    sessions = manager.get_sessions_in_range(start_date=start_date, end_date=end_date)

    if not sessions:
        if start_date or end_date:
            date_desc = f"from {start_date}" if start_date else ""
            date_desc += f" to {end_date}" if end_date else ""
            print(f"No sessions found {date_desc}")
        else:
            print("No sessions found")
        return

    if not start_date and not end_date:
        date_desc = "all"
    elif start_date and not end_date:
        date_desc = f"from {start_date}"
    elif not start_date and end_date:
        date_desc = f"to {end_date}"
    else:
        date_desc = f"from {start_date} to {end_date}"

    print(f"Status for sessions {date_desc}:\n")
    print(f"{'Date':<12} {'Participant':<15} {'DJ':<4} {'Backup':<8} {'Verified':<10} {'Safe':<6} {'Size (GB)':<10}")
    print("=" * 80)

    total_size = 0
    safe_count = 0

    for participant_id, sess_date in sessions:
        dj_imported = manager.check_datajoint_imported(participant_id, sess_date)
        backup_exists = manager.backup_exists(participant_id, sess_date)

        if backup_exists:
            verification = manager.verify_backup(participant_id, sess_date)
            verified = verification['verified']
            size_gb = verification['total_size'] / 1e9
            total_size += size_gb
        else:
            verified = False
            size_gb = 0

        safe, _ = manager.is_safe_to_delete(participant_id, sess_date)
        if safe:
            safe_count += 1

        dj_mark = '✓' if dj_imported else '✗'
        backup_mark = '✓' if backup_exists else '✗'
        verified_mark = '✓' if verified else '✗'
        safe_mark = '✓' if safe else '✗'

        print(f"{sess_date:<12} {participant_id:<15} {dj_mark:<4} {backup_mark:<8} {verified_mark:<10} {safe_mark:<6} {size_gb:>9.2f}")

    print("=" * 80)
    print(f"Total: {len(sessions)} sessions, {safe_count} safe to delete, {total_size:.2f} GB backed up")


def main():
    parser = argparse.ArgumentParser(description="Backup sync tool for MultiCameraTracking")
    parser.add_argument('--config', help='Path to backup_config.yaml')
    parser.add_argument('--start-date', help='Start date YYYYMMDD (for sync operations)')
    parser.add_argument('--end-date', help='End date YYYYMMDD (for sync operations)')
    parser.add_argument('--session', nargs=2, metavar=('PARTICIPANT', 'DATE'))
    parser.add_argument('--verify', nargs=2, metavar=('PARTICIPANT', 'DATE'))
    parser.add_argument('--status', nargs='+', metavar='PARTICIPANT_OR_DATE', help='Show status for participant+date or all sessions on date')
    parser.add_argument('--status-range', nargs='*', metavar='DATE', help='Show status for date range: --status-range [START] [END] (no args = all sessions)')
    parser.add_argument('--safe-delete', nargs=2, metavar=('PARTICIPANT', 'DATE'))
    parser.add_argument('--dry-run', action='store_true', help='Test without actually copying')
    parser.add_argument('--force', action='store_true', help='Force deletion even if not safe')

    args = parser.parse_args()

    db = get_db()
    manager = BackupManager(db, args.config)

    if args.session:
        sync_single_session(args, manager)
    elif args.start_date:
        sync_multiple_sessions(args, manager)
    elif args.verify:
        verify_backup(args, manager)
    elif args.status:
        show_status(args, manager)
    elif args.status_range is not None:
        show_status_range(args, manager)
    elif args.safe_delete:
        safe_delete_session(args, manager)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
