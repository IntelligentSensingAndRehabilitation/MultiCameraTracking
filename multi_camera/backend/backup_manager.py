import subprocess
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from sqlalchemy.orm import Session
from .recording_db import Participant, Session as SessionModel, Recording, Imported


class BackupConfig:
    """Load and validate backup configuration from YAML"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "backup_config.yaml"

        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Backup config not found at {self.config_path}. "
                f"Copy backup_config.example.yaml to backup_config.yaml and customize."
            )

        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

    def get_source_path(self, participant_id: str, session_date: str) -> Path:
        """Build source path from template"""
        base = self.config['backup']['source']['base_path']
        structure = self.config['backup']['source']['path_structure']
        path = structure.format(participant_id=participant_id, session_date=session_date)
        return Path(base) / path

    def get_destination_path(self, participant_id: str, session_date: str, dest_name: Optional[str] = None) -> Path:
        """Build destination path from template"""
        dest = self.config['backup']['destinations'][0]
        base = dest['base_path']
        structure = dest['path_structure']
        path = structure.format(participant_id=participant_id, session_date=session_date)
        return Path(base) / path

    def should_sync_participant(self, participant_id: str) -> bool:
        """Check if participant matches filter rules"""
        prefixes = self.config['backup']['filters']['participant_prefixes']
        return any(participant_id.startswith(p) for p in prefixes)

    def get_destination_config(self, dest_name: Optional[str] = None) -> dict:
        """Get destination configuration"""
        return self.config['backup']['destinations'][0]


class BackupManager:
    """Manages backup operations with verification"""

    def __init__(self, db: Session, config_path: Optional[str] = None):
        self.db = db
        self.config = BackupConfig(config_path)

    def verify_mount(self, destination_name: Optional[str] = None) -> Tuple[bool, str]:
        """Check if backup destination is mounted"""
        dest = self.config.get_destination_config(destination_name)
        mount_point = dest['mount_point']

        result = subprocess.run(
            ['mountpoint', '-q', mount_point],
            capture_output=True
        )

        if result.returncode == 0:
            if Path(dest['base_path']).exists():
                return True, f"{mount_point} mounted and accessible"
            else:
                return False, f"{dest['base_path']} not accessible"
        else:
            return False, f"{mount_point} not mounted"

    def sync_session(self, participant_id: str, session_date: str,
                     destination_name: Optional[str] = None, dry_run: bool = False) -> Dict:
        """
        Sync a single session to backup destination

        Returns dict with success status, stats, and verification results
        """
        mount_ok, mount_msg = self.verify_mount(destination_name)
        if not mount_ok:
            return {'success': False, 'error': mount_msg}

        if not self.config.should_sync_participant(participant_id):
            return {'success': False, 'error': f'Participant {participant_id} not in allowed list'}

        source = self.config.get_source_path(participant_id, session_date)
        dest = self.config.get_destination_path(participant_id, session_date, destination_name)

        if not source.exists():
            return {'success': False, 'error': f'Source path not found: {source}'}

        dest.mkdir(parents=True, exist_ok=True)

        rsync_flags = self.config.config['backup']['rsync']['flags'].split()
        cmd = ['rsync'] + rsync_flags + [f"{source}/", f"{dest}/"]

        if dry_run:
            cmd.append('--dry-run')

        start_time = datetime.now()
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = (datetime.now() - start_time).total_seconds()

        if result.returncode != 0:
            return {
                'success': False,
                'error': f'rsync failed with code {result.returncode}: {result.stderr}'
            }

        stats = self._parse_rsync_output(result.stdout)

        verification = {}
        if self.config.config['backup']['verification']['verify_after_sync'] and not dry_run:
            verification = self.verify_backup(participant_id, session_date, destination_name)

        return {
            'success': True,
            'files_synced': stats.get('files_transferred', 0),
            'bytes_transferred': stats.get('total_size', 0),
            'duration_seconds': duration,
            'verification': verification
        }

    def verify_backup(self, participant_id: str, session_date: str,
                     destination_name: Optional[str] = None) -> Dict:
        """
        Verify backup completeness using file counts and sizes

        Returns verification results dict
        """
        source = self.config.get_source_path(participant_id, session_date)
        dest = self.config.get_destination_path(participant_id, session_date, destination_name)

        source_files = list(source.rglob('*'))
        source_files = [f for f in source_files if f.is_file()]

        dest_files = list(dest.rglob('*'))
        dest_files = [f for f in dest_files if f.is_file()]

        file_count_match = len(source_files) == len(dest_files)

        source_names = {f.relative_to(source) for f in source_files}
        dest_names = {f.relative_to(dest) for f in dest_files}
        missing_files = list(source_names - dest_names)

        size_match = True
        total_size = 0
        for src_file in source_files:
            rel_path = src_file.relative_to(source)
            dest_file = dest / rel_path
            src_size = src_file.stat().st_size
            total_size += src_size

            if dest_file.exists():
                if src_size != dest_file.stat().st_size:
                    size_match = False
                    break

        verified = file_count_match and size_match and len(missing_files) == 0

        return {
            'verified': verified,
            'file_count_match': file_count_match,
            'size_match': size_match,
            'missing_files': [str(f) for f in missing_files],
            'source_file_count': len(source_files),
            'dest_file_count': len(dest_files),
            'total_size': total_size
        }

    def backup_exists(self, participant_id: str, session_date: str) -> bool:
        """Check if backup destination has files for this session"""
        dest = self.config.get_destination_path(participant_id, session_date)
        return dest.exists() and any(dest.iterdir())

    def check_datajoint_imported(self, participant_id: str, session_date: str) -> bool:
        """Check if session has been imported to DataJoint"""
        participant = self.db.query(Participant).filter(Participant.name == participant_id).first()
        if not participant:
            return False

        if isinstance(session_date, str):
            session_date = datetime.strptime(session_date, "%Y%m%d").date()

        session = self.db.query(SessionModel).filter(
            SessionModel.participant_id == participant.id,
            SessionModel.session_date == session_date
        ).first()

        if not session:
            return False

        imported = self.db.query(Imported).filter(Imported.session_id == session.id).first()
        return imported is not None

    def is_safe_to_delete(self, participant_id: str, session_date: str) -> Tuple[bool, str]:
        """
        Check if session is safe to delete

        Returns (safe, reason)
        """
        dj_imported = self.check_datajoint_imported(participant_id, session_date)
        if not dj_imported:
            return False, "Not imported to DataJoint"

        if not self.backup_exists(participant_id, session_date):
            return False, "No backup found"

        verification = self.verify_backup(participant_id, session_date)
        if not verification['verified']:
            return False, f"Backup verification failed"

        return True, "Safe to delete"

    def get_sessions_in_range(self, start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> List[Tuple[str, str]]:
        """
        Get all participant sessions in a date range

        Returns list of (participant_id, session_date) tuples
        """
        query = self.db.query(SessionModel)

        if start_date:
            if isinstance(start_date, str):
                start_date_obj = datetime.strptime(start_date, "%Y%m%d").date()
            else:
                start_date_obj = start_date
            query = query.filter(SessionModel.session_date >= start_date_obj)

        if end_date:
            if isinstance(end_date, str):
                end_date_obj = datetime.strptime(end_date, "%Y%m%d").date()
            else:
                end_date_obj = end_date
            query = query.filter(SessionModel.session_date <= end_date_obj)

        sessions = query.order_by(SessionModel.session_date.desc()).all()

        results = []
        for session in sessions:
            participant = self.db.query(Participant).filter(
                Participant.id == session.participant_id
            ).first()
            if participant and self.config.should_sync_participant(participant.name):
                results.append((participant.name, session.session_date.strftime("%Y%m%d")))

        return results

    def sync_multiple_sessions(self, start_date: str, end_date: Optional[str] = None,
                               participant_filter: Optional[str] = None,
                               dry_run: bool = False) -> Dict:
        """
        Batch sync multiple sessions

        Returns summary of results
        """
        if isinstance(start_date, str):
            start_date_obj = datetime.strptime(start_date, "%Y%m%d").date()
        else:
            start_date_obj = start_date

        query = self.db.query(SessionModel).filter(SessionModel.session_date >= start_date_obj)

        if end_date:
            if isinstance(end_date, str):
                end_date_obj = datetime.strptime(end_date, "%Y%m%d").date()
            else:
                end_date_obj = end_date
            query = query.filter(SessionModel.session_date <= end_date_obj)

        if participant_filter:
            participant = self.db.query(Participant).filter(Participant.name == participant_filter).first()
            if participant:
                query = query.filter(SessionModel.participant_id == participant.id)

        sessions = query.all()

        results = []
        succeeded = 0
        failed = 0

        for session in sessions:
            participant = self.db.query(Participant).filter(Participant.id == session.participant_id).first()
            participant_id = participant.name
            session_date = session.session_date.strftime("%Y%m%d")

            if not self.config.should_sync_participant(participant_id):
                continue

            print(f"Syncing {participant_id}/{session_date}...")
            result = self.sync_session(participant_id, session_date, dry_run=dry_run)
            results.append({
                'participant_id': participant_id,
                'session_date': session_date,
                **result
            })

            if result['success']:
                succeeded += 1
            else:
                failed += 1

        return {
            'sessions_processed': len(results),
            'sessions_succeeded': succeeded,
            'sessions_failed': failed,
            'results': results
        }

    def _parse_rsync_output(self, output: str) -> Dict:
        """Parse rsync stdout for statistics"""
        stats = {}
        for line in output.split('\n'):
            if 'total size is' in line:
                parts = line.split()
                if len(parts) >= 4:
                    stats['total_size'] = int(parts[3].replace(',', ''))
        return stats
