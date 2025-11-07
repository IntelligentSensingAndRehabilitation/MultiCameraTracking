# MultiCameraTracking Backup & Storage Management System

## Overview

Comprehensive backup and storage management system for multi-camera acquisition data with verification, tracking, and safe deletion workflows.

The system is designed to:
- **Sync** recordings from local storage (`/data`) to network storage (`/mnt/cottonlab`)
- **Verify** backups are complete before allowing deletion
- **Check** DataJoint import status (both SQLite flag AND actual data in tables)
- **Safely delete** local files after triple verification

## Architecture

### Design Philosophy

**Host-Based Operations** (Shell Script):
- All filesystem operations (rsync, verification, deletion, mount checks)
- Fast, reliable, no Docker overhead
- No mount-point issues or stale mount hangs

**Docker-Based Operations** (Python):
- Only database queries (DataJoint imports, session lookups)
- Leverages complex DataJoint environment
- Isolated from filesystem concerns

### Key Components

**`backup_manager.sh`** (Main CLI - runs on host)
- Handles all user interactions
- Executes rsync natively on host
- Calls Docker only for database queries
- Provides colored, user-friendly output

**`backup_db_queries.py`** (Database queries - runs in Docker)
- Checks DataJoint import status (SQLite + all DataJoint tables)
- Retrieves session lists from database
- NO filesystem operations

**`backup_config.yaml`** (Configuration)
- External YAML config (gitignored)
- User-customized paths and settings
- Flexible path structures (supports inverted layouts)

## Data Flow

```
1. Record Session
   → /data/{participant}/{date}/

2. Push to DataJoint (via GUI button)
   → SQLite "Imported" flag set
   → Data in DataJoint tables: Subject, Session, Recording,
     MultiCameraRecording, SingleCameraVideo

3. Sync to Network (via backup_manager.sh)
   → rsync: /data/ → /mnt/cottonlab/
   → Automatic verification (file counts)

4. Verify Backup (optional, thorough check)
   → File-by-file size comparison
   → Identifies missing/mismatched files

5. Safe Delete (via backup_manager.sh)
   → Check: DataJoint ✓ + Backup ✓ + Verified ✓
   → Require "DELETE" confirmation
   → Delete local files from /data
```

## Configuration

### Setup

1. Copy example config:
```bash
cd packages/MultiCameraTracking
cp backup_config.example.yaml backup_config.yaml
```

2. Edit `backup_config.yaml` with your environment settings

3. Ensure script is executable:
```bash
chmod +x backup_manager.sh
```

### Configuration Structure

```yaml
backup:
  destinations:
    - name: "CottonLab Network"
      mount_point: "/mnt/CottonLab"
      base_path: "/mnt/CottonLab/mobile_system_data"
      # Path structure at destination (can differ from source)
      path_structure: "{session_date}/{participant_id}"
      enabled: true
      verify_mount: true

  source:
    base_path: "/data"
    path_structure: "{participant_id}/{session_date}"

  filters:
    # Only sync participants with these prefixes
    participant_prefixes: ["m", "p", "b", "s", "TR", "TF"]
    exclude_patterns: ["*.tmp", "*.lock"]

  rsync:
    flags: "-avzh"

  verification:
    verify_after_sync: true
    verify_method: "size"  # File size + count verification
```

### Path Structure Examples

**Source** (local): `/data/p001/20250104/`
**Destination** (network): `/mnt/CottonLab/mobile_system_data/20250104/p001/`

Note: The path structures can be inverted or customized using the `path_structure` template variables.

## CLI Usage

All commands should be run from the `packages/MultiCameraTracking/` directory on the **host system** (not inside Docker).

### Commands Reference

#### 1. Sync Single Session

Copies a single session from local to network storage with automatic verification.

```bash
./backup_manager.sh sync <participant_id> <session_date> [--dry-run]
```

**Examples:**
```bash
./backup_manager.sh sync p001 20250104
./backup_manager.sh sync p001 20250104 --dry-run  # Test without copying
```

**What it does:**
- Verifies mount point is accessible
- Executes rsync with progress indicator
- Verifies file counts match after sync
- Reports success/failure

---

#### 2. Batch Sync Multiple Sessions

Syncs all sessions in a date range in one operation.

```bash
./backup_manager.sh batch <start_date> [end_date] [--dry-run]
```

**Examples:**
```bash
./backup_manager.sh batch 20250104           # Single day
./backup_manager.sh batch 20250101 20250131  # Date range
./backup_manager.sh batch 20250101 20250131 --dry-run
```

**What it does:**
- Queries database for all sessions in range
- Skips sessions that don't exist locally
- Skips sessions already backed up and verified
- Shows progress for each session
- Displays summary (succeeded/failed/skipped)

---

#### 3. Check Status (Single Session)

Shows detailed status for a specific session.

```bash
./backup_manager.sh status <participant_id> <session_date>
```

**Example:**
```bash
./backup_manager.sh status p001 20250104
```

**Output includes:**
- DataJoint import status (SQLite + all DataJoint tables)
- Source location, file count, and size
- Backup location, file count, and size
- File count comparison
- Safe-to-delete determination

---

#### 4. Check Status Range (Multiple Sessions)

Shows overview table of all sessions with local data.

```bash
./backup_manager.sh status-range [--start-date YYYYMMDD] [--end-date YYYYMMDD]
```

**Examples:**
```bash
# All sessions with local data
./backup_manager.sh status-range

# Sessions from Jan 1st onwards
./backup_manager.sh status-range --start-date 20250101

# Sessions up to Jan 31st
./backup_manager.sh status-range --end-date 20250131

# Sessions in January
./backup_manager.sh status-range --start-date 20250101 --end-date 20250131
```

**Output columns:**
- **Date**: Session date
- **Participant**: Participant ID
- **Videos**: Number of .mp4 video files in the session
- **DJ**: DataJoint imported (✓/✗)
- **Backup**: Backup exists on network (✓/✗)
- **Verified**: File counts match (✓/✗)
- **Safe**: Safe to delete (all checks passed) (✓/✗)

**Note:** Only shows sessions that exist locally in `/data`. Sessions already deleted won't appear.

---

#### 5. Verify Backup Integrity

Performs thorough file-by-file verification of a backup.

```bash
./backup_manager.sh verify <participant_id> <session_date>
```

**Example:**
```bash
./backup_manager.sh verify p001 20250104
```

**What it checks:**
- DataJoint import status
- Mount accessibility
- Source and destination both exist
- File counts match
- **Every file size matches** (progress indicator shown)
- Identifies missing or mismatched files

**When to use:**
- Before deleting important sessions
- After network issues during sync
- Periodic integrity checks

---

#### 6. Safe Delete Session

Deletes local files after comprehensive safety checks.

```bash
./backup_manager.sh delete <participant_id> <session_date> [--dry-run]
```

**Examples:**
```bash
./backup_manager.sh delete p001 20250104
./backup_manager.sh delete p001 20250104 --dry-run  # Test safety checks without deleting
```

**Options:**
- `--dry-run`: Perform all safety checks and show what would be deleted without actually deleting files

**Safety checks performed:**
1. **DataJoint verification** (checks both SQLite Imported flag AND actual data in all DataJoint tables: Subject, Session, Recording, MultiCameraRecording, SingleCameraVideo)
2. **Backup exists** on network storage
3. **Backup integrity** verified (file counts + sample file sizes match)

**Deletion workflow:**
1. Performs all safety checks
2. Shows deletion summary (path, file count, size, where data is preserved)
3. In normal mode: Requires typing `DELETE` (in capitals) to confirm
4. In dry-run mode: Skips confirmation and deletion, shows what would be deleted
5. Executes deletion (`rm -rf`) or reports dry-run results
6. Verifies deletion succeeded (normal mode only)

**Where data is preserved after deletion:**
- DataJoint database tables (all video metadata and processing results)
- Network backup at `/mnt/CottonLab/...`

**When to use --dry-run:**
- Test deletion safety checks before committing
- Verify that all safety conditions are met
- Preview deletion summary without risk

---

#### 7. Bulk Delete Sessions

Deletes all sessions that are safe to delete (DataJoint imported + backup verified).

```bash
./backup_manager.sh bulk-delete [--start-date YYYYMMDD] [--end-date YYYYMMDD] [--dry-run]
```

**Examples:**
```bash
# Delete all safe-to-delete sessions
./backup_manager.sh bulk-delete

# Delete safe sessions from January 2025
./backup_manager.sh bulk-delete --start-date 20250101 --end-date 20250131

# Preview what would be deleted (dry-run)
./backup_manager.sh bulk-delete --dry-run

# Preview with date filter
./backup_manager.sh bulk-delete --start-date 20250101 --end-date 20250131 --dry-run
```

**Options:**
- `--start-date YYYYMMDD`: Only consider sessions from this date onwards (optional)
- `--end-date YYYYMMDD`: Only consider sessions up to this date (optional)
- `--dry-run`: Show what would be deleted without actually deleting files

**What it does:**
1. Queries database for all sessions in date range (or all sessions if no dates specified)
2. Checks each session against safety criteria:
   - DataJoint fully imported (SQLite + all DataJoint tables)
   - Backup exists on network storage
   - Backup verified (file counts match)
3. Displays list of sessions that would be deleted with sizes
4. Shows total space to be freed
5. In normal mode: Requires typing `DELETE ALL` (exactly) to confirm
6. In dry-run mode: Skips confirmation and deletion, shows preview only
7. Deletes each session and reports success/failure
8. Displays summary of results

**When to use:**
- Cleanup after bulk syncing multiple sessions
- Free up disk space when many sessions are safely backed up
- Routine maintenance to keep local storage clean
- Use `--dry-run` first to preview what will be deleted

**Safety features:**
- Same safety checks as single `delete` command
- Lists all sessions before deletion
- Requires exact confirmation phrase (`DELETE ALL`)
- Reports individual success/failure for each session
- Can be filtered by date range to limit scope

---

## Column Explanations

### Status-Range Output Columns

- **Date**: Session date in YYYYMMDD format

- **Participant**: Participant ID

- **Videos**: Number of .mp4 video files in the session directory
  - Useful for quick sanity check (typical sessions have 4-8 camera videos)

- **DJ (DataJoint)**: ✓ if session is fully imported to DataJoint
  - Checks SQLite `Imported` flag
  - **AND** checks actual DataJoint tables contain data:
    - `Subject` table has participant
    - `Session` table has session entry
    - `Recording` table has recording links
    - `MultiCameraRecording` table has recording metadata
    - `SingleCameraVideo` table has camera video entries

- **Backup**: ✓ if backup directory exists on network storage

- **Verified**: ✓ if file count in source matches file count in backup (and > 0 files)
  - This is a **quick check** (file count only)
  - For thorough verification (file-by-file size check), use `verify` command

- **Safe**: ✓ if all three conditions are met:
  - DJ = ✓ (fully imported to DataJoint)
  - Backup = ✓ (backup exists)
  - Verified = ✓ (file counts match)

## Troubleshooting

### Mount Point Issues

**Error: "Mount point not accessible (may be stale)"**

Solution:
```bash
sudo umount /mnt/CottonLab
sudo mount /mnt/CottonLab
```

### Empty Directory Errors

**Error: "Source directory is empty"**

The script now properly handles empty directories and will report this clearly rather than causing integer expression errors.

### DataJoint Not Imported

**Error: "Session not fully imported to DataJoint"**

This means either:
- Session not in SQLite database
- Session not pushed to DataJoint (use GUI "push_to_datajoint" button)
- Data missing from one or more DataJoint tables

Check what's missing:
```bash
./backup_manager.sh status p001 20250104
```

### Verification Failed

**Error: "File count mismatch" or "Size mismatches found"**

Run thorough verification to see details:
```bash
./backup_manager.sh verify p001 20250104
```

This will show which files are missing or have wrong sizes.

## Safety Features

### Triple Verification for Deletion

1. **Database verification**: Checks both SQLite tracking AND actual DataJoint table data
2. **Filesystem verification**: Confirms backup exists and file counts match
3. **Sample integrity check**: Verifies 10 random file sizes match

### Confirmation Required

Deletion requires typing `DELETE` (in capitals) - prevents accidental deletion.

### Dry-Run Mode

The `--dry-run` flag allows you to test all safety checks and preview deletion without actually removing files.

## Development Notes

### Architecture Decisions

**Why shell script instead of Python for filesystem operations?**
- Mount operations are OS-level (best handled by shell commands)
- Avoids Python `pathlib` hanging on stale mounts
- Faster rsync execution (native, not subprocess)
- Simpler deployment (no Python dependencies on host)

**Why keep Docker for database queries?**
- Complex DataJoint environment with specific versions
- Database connection config lives in container
- Schema access requires DataJoint table definitions
- Isolation of DB operations from filesystem

### No Backup Tracking in Database

User decision: Filesystem + DataJoint are the sources of truth. No separate backup tracking table needed.

Benefits:
- Simpler system
- Less state to manage
- Verification always checks actual files
- No database sync issues

## See Also

- [backup_config.example.yaml](backup_config.example.yaml) - Configuration template
- [backup_manager.sh](backup_manager.sh) - Main backup CLI script
- [scripts/backup_db_queries.py](scripts/backup_db_queries.py) - Database query tool
