#!/usr/bin/env python3
"""
Backup sync tool for MultiCameraTracking sessions

Usage:
  backup_sync.py --start-date 20250101 [--end-date 20250131] [--dry-run]
  backup_sync.py --session <participant_id> <session_date> [--dry-run]
  backup_sync.py --verify <participant_id> <session_date>
  backup_sync.py --status <participant_id> <session_date>
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
    """Show comprehensive status for a session"""
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


def main():
    parser = argparse.ArgumentParser(description="Backup sync tool for MultiCameraTracking")
    parser.add_argument('--config', help='Path to backup_config.yaml')
    parser.add_argument('--start-date', help='Start date YYYYMMDD')
    parser.add_argument('--end-date', help='End date YYYYMMDD')
    parser.add_argument('--session', nargs=2, metavar=('PARTICIPANT', 'DATE'))
    parser.add_argument('--verify', nargs=2, metavar=('PARTICIPANT', 'DATE'))
    parser.add_argument('--status', nargs=2, metavar=('PARTICIPANT', 'DATE'))
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
    elif args.safe_delete:
        safe_delete_session(args, manager)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
