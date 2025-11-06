#!/bin/bash

################################################################################
# MultiCameraTracking Backup Manager
#
# Purpose: Manage backup operations for multi-camera acquisition data
# Usage:
#   ./backup_manager.sh sync <participant_id> <session_date> [--dry-run]
#   ./backup_manager.sh batch <start_date> [end_date] [--dry-run]
#   ./backup_manager.sh status <participant_id> <session_date>
#   ./backup_manager.sh status <session_date>
#   ./backup_manager.sh status-range [start_date] [end_date]
#   ./backup_manager.sh verify <participant_id> <session_date>
#   ./backup_manager.sh delete <participant_id> <session_date> [--force]
#
# This script should be run from the MultiCameraTracking repository root
# on the HOST system (not inside container)
################################################################################

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Print error message
print_error() {
    local message="$1"
    echo -e "${RED}[ERROR]${NC} ${message}"
}

# Print success message
print_success() {
    local message="$1"
    echo -e "${GREEN}[OK]${NC} ${message}"
}

# Print info message
print_info() {
    local message="$1"
    echo "${message}"
}

# Check if we're in the right directory
check_directory() {
    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found"
        echo "Please run this script from the MultiCameraTracking repository root"
        exit 1
    fi

    if [ ! -f "backup_config.yaml" ]; then
        print_error "backup_config.yaml not found"
        echo "Copy backup_config.example.yaml to backup_config.yaml and customize it"
        exit 1
    fi
}

# Parse backup config using Python
parse_config() {
    python3 -c "
import yaml
with open('backup_config.yaml') as f:
    config = yaml.safe_load(f)
print(config['backup']['source']['base_path'])
print(config['backup']['destinations'][0]['base_path'])
print(config['backup']['destinations'][0]['mount_point'])
print(config['backup']['source']['path_structure'])
print(config['backup']['destinations'][0]['path_structure'])
print(config['backup']['rsync']['flags'])
"
}

# Get config values
get_config_values() {
    local config_output=$(parse_config)
    SOURCE_BASE=$(echo "$config_output" | sed -n '1p')
    DEST_BASE=$(echo "$config_output" | sed -n '2p')
    MOUNT_POINT=$(echo "$config_output" | sed -n '3p')
    SOURCE_STRUCTURE=$(echo "$config_output" | sed -n '4p')
    DEST_STRUCTURE=$(echo "$config_output" | sed -n '5p')
    RSYNC_FLAGS=$(echo "$config_output" | sed -n '6p')
}

# Build source path from template
get_source_path() {
    local participant_id="$1"
    local session_date="$2"
    local path="${SOURCE_STRUCTURE//\{participant_id\}/$participant_id}"
    path="${path//\{session_date\}/$session_date}"
    echo "${SOURCE_BASE}/${path}"
}

# Build destination path from template
get_dest_path() {
    local participant_id="$1"
    local session_date="$2"
    local path="${DEST_STRUCTURE//\{participant_id\}/$participant_id}"
    path="${path//\{session_date\}/$session_date}"
    echo "${DEST_BASE}/${path}"
}

# Verify mount point is accessible
verify_mount() {
    if [ -z "$MOUNT_POINT" ]; then
        print_error "Could not read mount_point from backup_config.yaml"
        return 1
    fi

    # Check if mount point exists and is mounted
    if ! mountpoint -q "$MOUNT_POINT" 2>/dev/null; then
        print_error "Mount point $MOUNT_POINT is not mounted"
        echo "Please ensure your network storage is mounted before running backup operations"
        return 1
    fi

    # Verify it's accessible (not stale)
    if ! timeout 2 ls "$MOUNT_POINT" >/dev/null 2>&1; then
        print_error "Mount point $MOUNT_POINT is not accessible (may be stale)"
        echo "You may need to unmount and remount: sudo umount $MOUNT_POINT && sudo mount $MOUNT_POINT"
        return 1
    fi

    return 0
}

# Query database via Docker (only for DB operations)
query_db() {
    docker compose run --rm --entrypoint python3 backup /Mocap/scripts/backup_db_queries.py "$@"
}

# Check if session is imported to DataJoint
check_datajoint_imported() {
    local participant_id="$1"
    local session_date="$2"
    query_db --check-datajoint "$participant_id" "$session_date" >/dev/null 2>&1
    return $?
}

# Get all sessions from database
get_sessions_from_db() {
    query_db --get-sessions "$@"
}

# Perform rsync (on host)
do_rsync() {
    local source="$1"
    local dest="$2"
    local dry_run="$3"
    local show_progress="${4:-true}"  # Show progress by default

    if [ ! -d "$source" ]; then
        print_error "Source path not found: $source"
        return 1
    fi

    local cmd="rsync $RSYNC_FLAGS"
    [ "$dry_run" = "true" ] && cmd="$cmd --dry-run"

    # Add progress flag if showing progress
    if [ "$show_progress" = "true" ]; then
        cmd="$cmd --progress"
    fi

    cmd="$cmd \"${source}/\" \"${dest}/\""

    if [ "$show_progress" = "true" ]; then
        print_info "Syncing files..."
    fi

    eval $cmd
    return $?
}

# Main command dispatcher
main() {
    check_directory
    get_config_values  # Load config into global variables

    if [ $# -eq 0 ]; then
        echo "MultiCameraTracking Backup Manager"
        echo ""
        echo "Usage:"
        echo "  $0 sync <participant_id> <session_date> [--dry-run]"
        echo "      Sync a single session to backup destination"
        echo ""
        echo "  $0 batch <start_date> [end_date] [--dry-run]"
        echo "      Batch sync sessions in date range (YYYYMMDD format)"
        echo ""
        echo "  $0 status <participant_id> <session_date>"
        echo "      Show status for a specific session"
        echo ""
        echo "  $0 status <session_date>"
        echo "      Show status for all sessions on a specific date"
        echo ""
        echo "  $0 status-range [--start-date YYYYMMDD] [--end-date YYYYMMDD]"
        echo "      Show status for sessions in date range"
        echo "      --start-date: Show sessions from this date onwards (optional)"
        echo "      --end-date: Show sessions up to this date (optional)"
        echo "      No args: Show all sessions with local data"
        echo ""
        echo "  $0 verify <participant_id> <session_date>"
        echo "      Verify backup integrity for a session"
        echo ""
        echo "  $0 delete <participant_id> <session_date> [--force]"
        echo "      Safely delete local session after verification"
        echo ""
        echo "Examples:"
        echo "  $0 sync p001 20250104"
        echo "  $0 batch 20250101 20250131"
        echo "  $0 status p001 20250104"
        echo "  $0 status 20250104"
        echo "  $0 status-range"
        echo "  $0 status-range --start-date 20250101"
        echo "  $0 status-range --end-date 20250131"
        echo "  $0 status-range --start-date 20250101 --end-date 20250131"
        echo "  $0 delete p001 20250104"
        exit 1
    fi

    local command="$1"
    shift

    case "$command" in
        sync)
            if [ $# -lt 2 ]; then
                print_error "Usage: $0 sync <participant_id> <session_date> [--dry-run]"
                exit 1
            fi

            # Verify mount before syncing
            if ! verify_mount; then
                exit 1
            fi

            local participant_id="$1"
            local session_date="$2"
            local dry_run="false"
            shift 2
            [[ "$1" == "--dry-run" ]] && dry_run="true"

            print_info "Syncing ${participant_id}/${session_date}..."

            local source=$(get_source_path "$participant_id" "$session_date")
            local dest=$(get_dest_path "$participant_id" "$session_date")

            # Perform rsync on host
            if do_rsync "$source" "$dest" "$dry_run"; then
                print_success "Sync completed"

                # Verify if not dry run
                if [ "$dry_run" = "false" ]; then
                    print_info "Verifying backup..."

                    # Count files in source and destination
                    local source_count=$(find "$source" -type f 2>/dev/null | wc -l)
                    local dest_count=$(find "$dest" -type f 2>/dev/null | wc -l)

                    # Handle empty directories
                    source_count=${source_count:-0}
                    dest_count=${dest_count:-0}

                    if [ "$source_count" -eq 0 ]; then
                        print_error "Source directory is empty - nothing to sync!"
                        exit 1
                    fi

                    if [ "$source_count" -eq "$dest_count" ]; then
                        print_success "Verification passed: $source_count files synced"

                        # Calculate sizes
                        local source_size=$(du -sh "$source" | cut -f1)
                        local dest_size=$(du -sh "$dest" | cut -f1)
                        print_info "Source size: $source_size"
                        print_info "Backup size: $dest_size"
                    else
                        print_error "Verification FAILED: file count mismatch"
                        echo "  Source: $source_count files"
                        echo "  Backup: $dest_count files"
                        echo ""
                        echo "Run './backup_manager.sh verify $participant_id $session_date' for detailed analysis"
                        exit 1
                    fi
                fi
            else
                print_error "Sync failed"
                exit 1
            fi
            ;;

        batch)
            if [ $# -lt 1 ]; then
                print_error "Usage: $0 batch <start_date> [end_date] [--dry-run]"
                exit 1
            fi

            # Verify mount before batch sync
            if ! verify_mount; then
                exit 1
            fi

            local start_date="$1"
            shift
            local end_date=""
            local dry_run="false"

            # Check if next arg is a date or a flag
            if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]{8}$ ]]; then
                end_date="$1"
                shift
            fi

            # Check for dry-run flag
            [[ "$1" == "--dry-run" ]] && dry_run="true"

            echo -e "\n${BOLD}Batch sync sessions${NC}"
            if [ -n "$end_date" ]; then
                echo "Date range: $start_date to $end_date"
            else
                echo "Date: $start_date"
            fi
            [ "$dry_run" = "true" ] && echo "Mode: DRY RUN (no changes will be made)"
            echo ""

            # Get sessions from database
            print_info "Querying sessions from database..."
            local sessions=$(get_sessions_from_db "$start_date" "$end_date")

            if [ -z "$sessions" ]; then
                print_error "No sessions found in date range"
                exit 1
            fi

            # Count total sessions
            local total=$(echo "$sessions" | wc -l)
            echo "Found $total session(s) to sync"
            echo ""

            # Initialize counters
            local succeeded=0
            local failed=0
            local skipped=0
            local current=0

            # Loop through sessions
            while IFS=$'\t' read -r participant_id session_date dj_imported; do
                current=$((current + 1))

                echo -e "${BOLD}[$current/$total]${NC} Processing ${participant_id}/${session_date}..."

                local source=$(get_source_path "$participant_id" "$session_date")
                local dest=$(get_dest_path "$participant_id" "$session_date")

                # Check if source exists
                if [ ! -d "$source" ]; then
                    print_error "  Source not found, skipping"
                    skipped=$((skipped + 1))
                    echo ""
                    continue
                fi

                # Check if already backed up
                if [ -d "$dest" ]; then
                    local source_count=$(find "$source" -type f 2>/dev/null | wc -l)
                    local dest_count=$(find "$dest" -type f 2>/dev/null | wc -l)
                    source_count=${source_count:-0}
                    dest_count=${dest_count:-0}
                    if [ "$source_count" -eq "$dest_count" ] && [ "$source_count" -gt 0 ]; then
                        print_info "  Already backed up and verified, skipping"
                        skipped=$((skipped + 1))
                        echo ""
                        continue
                    fi
                fi

                # Perform sync (hide rsync progress for batch, show summary instead)
                print_info "  Syncing..."
                if do_rsync "$source" "$dest" "$dry_run" "false" 2>&1 | grep -E '(sent|total size)' | tail -2; then
                    succeeded=$((succeeded + 1))
                    print_success "  Synced successfully"

                    # Quick verification if not dry run
                    if [ "$dry_run" = "false" ]; then
                        local source_count=$(find "$source" -type f 2>/dev/null | wc -l)
                        local dest_count=$(find "$dest" -type f 2>/dev/null | wc -l)
                        source_count=${source_count:-0}
                        dest_count=${dest_count:-0}
                        if [ "$source_count" -eq "$dest_count" ] && [ "$source_count" -gt 0 ]; then
                            print_info "  Verified: $source_count files"
                        else
                            print_error "  Warning: file count mismatch after sync"
                        fi
                    fi
                else
                    failed=$((failed + 1))
                    print_error "  Sync failed"
                fi

                echo ""
            done <<< "$sessions"

            # Show summary
            echo -e "${BOLD}═══════════════════════════════════════════${NC}"
            echo -e "${BOLD}Batch sync summary:${NC}"
            echo "  Total sessions: $total"
            echo "  Succeeded: $succeeded"
            echo "  Failed: $failed"
            echo "  Skipped: $skipped"
            echo -e "${BOLD}═══════════════════════════════════════════${NC}"

            if [ $failed -gt 0 ]; then
                exit 1
            fi
            ;;

        status)
            if [ $# -lt 2 ]; then
                print_error "Usage: $0 status <participant_id> <session_date>"
                exit 1
            fi

            local participant_id="$1"
            local session_date="$2"

            echo -e "\n${BOLD}Status for ${participant_id}/${session_date}:${NC}\n"

            # Check DataJoint import status (query DB via Docker)
            if check_datajoint_imported "$participant_id" "$session_date"; then
                print_success "DataJoint imported: ✓"
            else
                print_error "DataJoint imported: ✗"
            fi

            # Check if backup exists (on host filesystem)
            local source=$(get_source_path "$participant_id" "$session_date")
            local dest=$(get_dest_path "$participant_id" "$session_date")

            local source_count=0
            local dest_count=0

            if [ -d "$source" ]; then
                source_count=$(find "$source" -type f 2>/dev/null | wc -l)
                source_count=${source_count:-0}
                local source_size=$(du -sh "$source" | cut -f1)
                print_success "Source exists: $source ($source_count files, $source_size)"
            else
                print_error "Source not found: $source"
            fi

            if verify_mount >/dev/null 2>&1 && [ -d "$dest" ]; then
                dest_count=$(find "$dest" -type f 2>/dev/null | wc -l)
                dest_count=${dest_count:-0}
                local dest_size=$(du -sh "$dest" | cut -f1)
                print_success "Backup exists: $dest ($dest_count files, $dest_size)"

                # Check if counts match
                if [ "$source_count" -eq "$dest_count" ] && [ "$source_count" -gt 0 ]; then
                    print_success "File counts match"
                else
                    print_error "File count mismatch: source=$source_count, backup=$dest_count"
                fi
            else
                print_error "Backup not found or mount not accessible"
            fi

            # Safe to delete?
            if check_datajoint_imported "$participant_id" "$session_date" && \
               verify_mount >/dev/null 2>&1 && [ -d "$dest" ] && \
               [ "$source_count" -eq "$dest_count" ]; then
                echo -e "\n${GREEN}${BOLD}✓ Safe to delete local files${NC}"
            else
                echo -e "\n${RED}${BOLD}✗ NOT safe to delete${NC}"
            fi
            ;;

        status-range)
            # Get sessions from database and check backup status for each
            # Only show sessions that exist locally in the data directory
            echo -e "\n${BOLD}Querying sessions from database...${NC}\n"

            local start_date=""
            local end_date=""

            # Parse arguments for --start-date and --end-date flags
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --start-date)
                        start_date="$2"
                        shift 2
                        ;;
                    --end-date)
                        end_date="$2"
                        shift 2
                        ;;
                    *)
                        print_error "Unknown argument: $1"
                        echo "Usage: $0 status-range [--start-date YYYYMMDD] [--end-date YYYYMMDD]"
                        exit 1
                        ;;
                esac
            done

            printf "%-12s %-15s %-6s %-8s %-10s %-6s\n" "Date" "Participant" "DJ" "Backup" "Verified" "Safe"
            echo "================================================================================"

            local displayed_count=0

            while IFS=$'\t' read -r participant_id session_date dj_imported; do
                local source=$(get_source_path "$participant_id" "$session_date")

                # Skip sessions that don't exist locally
                if [ ! -d "$source" ]; then
                    continue
                fi

                displayed_count=$((displayed_count + 1))

                local dest=$(get_dest_path "$participant_id" "$session_date")

                # Check if backup exists
                local backup_exists="✗"
                local verified="✗"
                if verify_mount >/dev/null 2>&1 && [ -d "$dest" ]; then
                    backup_exists="✓"

                    # Quick verification - just check file counts
                    local source_count=$(find "$source" -type f 2>/dev/null | wc -l)
                    local dest_count=$(find "$dest" -type f 2>/dev/null | wc -l)
                    source_count=${source_count:-0}
                    dest_count=${dest_count:-0}
                    [ "$source_count" -eq "$dest_count" ] && [ "$source_count" -gt 0 ] && verified="✓"
                fi

                # Safe to delete?
                local safe="✗"
                [ "$dj_imported" = "True" ] && [ "$backup_exists" = "✓" ] && [ "$verified" = "✓" ] && safe="✓"

                local dj_mark=$([ "$dj_imported" = "True" ] && echo "✓" || echo "✗")

                printf "%-12s %-15s %-6s %-8s %-10s %-6s\n" \
                    "$session_date" "$participant_id" "$dj_mark" "$backup_exists" "$verified" "$safe"

            done < <(get_sessions_from_db "$start_date" "$end_date")

            if [ $displayed_count -eq 0 ]; then
                echo ""
                print_info "No sessions found locally in the data directory"
                if [ -n "$start_date" ]; then
                    echo "Date range: $start_date to ${end_date:-$start_date}"
                fi
            fi
            ;;

        verify)
            if [ $# -ne 2 ]; then
                print_error "Usage: $0 verify <participant_id> <session_date>"
                exit 1
            fi

            local participant_id="$1"
            local session_date="$2"

            echo -e "\n${BOLD}Verifying backup for ${participant_id}/${session_date}:${NC}\n"

            # Check DataJoint import status
            if check_datajoint_imported "$participant_id" "$session_date"; then
                print_success "DataJoint imported: ✓"
            else
                print_error "DataJoint imported: ✗ (session not in database)"
                echo -e "\n${RED}${BOLD}✗ Verification FAILED${NC}"
                exit 1
            fi

            # Verify mount before verification
            if ! verify_mount; then
                echo -e "\n${RED}${BOLD}✗ Verification FAILED${NC}"
                exit 1
            fi

            local source=$(get_source_path "$participant_id" "$session_date")
            local dest=$(get_dest_path "$participant_id" "$session_date")

            # Check source exists
            if [ ! -d "$source" ]; then
                print_error "Source directory not found: $source"
                echo -e "\n${RED}${BOLD}✗ Verification FAILED${NC}"
                exit 1
            fi

            # Check destination exists
            if [ ! -d "$dest" ]; then
                print_error "Backup directory not found: $dest"
                echo -e "\n${RED}${BOLD}✗ Verification FAILED${NC}"
                exit 1
            fi

            print_info "Comparing files..."

            # Get file lists with relative paths
            local source_files=$(cd "$source" && find . -type f | sort)
            local dest_files=$(cd "$dest" && find . -type f | sort)

            # Count files (handle empty case)
            local source_count=$(echo "$source_files" | grep -c . || echo "0")
            local dest_count=$(echo "$dest_files" | grep -c . || echo "0")

            print_info "Source files: $source_count"
            print_info "Backup files: $dest_count"

            # Check if empty
            if [ "$source_count" -eq 0 ]; then
                print_error "Source directory is empty!"
                echo -e "\n${RED}${BOLD}✗ Verification FAILED${NC}"
                exit 1
            fi

            # Check file count match
            if [ "$source_count" -ne "$dest_count" ]; then
                print_error "File count mismatch!"

                # Find missing files
                echo -e "\n${YELLOW}Files in source but not in backup:${NC}"
                comm -23 <(echo "$source_files") <(echo "$dest_files") | head -10
                [ $(comm -23 <(echo "$source_files") <(echo "$dest_files") | wc -l) -gt 10 ] && echo "... and more"

                echo -e "\n${YELLOW}Files in backup but not in source:${NC}"
                comm -13 <(echo "$source_files") <(echo "$dest_files") | head -10
                [ $(comm -13 <(echo "$source_files") <(echo "$dest_files") | wc -l) -gt 10 ] && echo "... and more"

                echo -e "\n${RED}${BOLD}✗ Verification FAILED${NC}"
                exit 1
            fi

            print_success "File counts match"

            # Compare file sizes
            print_info "Verifying file sizes..."
            local size_mismatches=0
            local files_checked=0

            while IFS= read -r file; do
                files_checked=$((files_checked + 1))

                # Progress indicator every 100 files
                if [ $((files_checked % 100)) -eq 0 ]; then
                    echo -ne "\rChecked $files_checked/$source_count files..."
                fi

                local source_size=$(stat -c %s "$source/$file" 2>/dev/null)
                local dest_size=$(stat -c %s "$dest/$file" 2>/dev/null)

                if [ "$source_size" != "$dest_size" ]; then
                    if [ $size_mismatches -eq 0 ]; then
                        echo -e "\n\n${YELLOW}Size mismatches found:${NC}"
                    fi
                    size_mismatches=$((size_mismatches + 1))
                    echo "  $file: source=$source_size bytes, backup=$dest_size bytes"

                    # Limit output
                    if [ $size_mismatches -ge 10 ]; then
                        echo "  ... and possibly more (stopping after 10)"
                        break
                    fi
                fi
            done <<< "$source_files"

            echo -e "\rChecked $files_checked files.                    "

            if [ $size_mismatches -gt 0 ]; then
                print_error "Found $size_mismatches file(s) with size mismatches"
                echo -e "\n${RED}${BOLD}✗ Verification FAILED${NC}"
                exit 1
            fi

            print_success "All file sizes match"

            # Calculate total sizes
            local source_size=$(du -sh "$source" | cut -f1)
            local dest_size=$(du -sh "$dest" | cut -f1)
            print_info "Source total size: $source_size"
            print_info "Backup total size: $dest_size"

            echo -e "\n${GREEN}${BOLD}✓ Verification PASSED${NC}"
            echo "  - DataJoint imported: ✓"
            echo "  - File count match: $source_count files"
            echo "  - All file sizes match: ✓"
            echo "  - Total size: $source_size"
            ;;

        delete)
            if [ $# -lt 2 ]; then
                print_error "Usage: $0 delete <participant_id> <session_date> [--force]"
                exit 1
            fi

            local participant_id="$1"
            local session_date="$2"
            local force_delete="false"
            shift 2
            [[ "$1" == "--force" ]] && force_delete="true"

            echo -e "\n${BOLD}Safety check for deleting ${participant_id}/${session_date}:${NC}\n"

            local source=$(get_source_path "$participant_id" "$session_date")
            local dest=$(get_dest_path "$participant_id" "$session_date")

            # Check source exists
            if [ ! -d "$source" ]; then
                print_error "Source directory not found: $source"
                echo "Nothing to delete."
                exit 1
            fi

            # 1. Check DataJoint import status (checks both SQLite flag AND actual DataJoint tables)
            print_info "Checking DataJoint database..."
            if ! check_datajoint_imported "$participant_id" "$session_date"; then
                if [ "$force_delete" = "false" ]; then
                    print_error "Session not fully imported to DataJoint"
                    echo ""
                    echo "This means either:"
                    echo "  - Session not marked as imported in SQLite database, OR"
                    echo "  - Data missing from DataJoint tables (Subject, Session, Recording, MultiCameraRecording, SingleCameraVideo)"
                    echo ""
                    echo "Cannot safely delete without verified DataJoint data."
                    echo "Use --force to override (NOT RECOMMENDED - may result in data loss)"
                    exit 1
                else
                    print_error "Session not fully imported to DataJoint (--force used, proceeding anyway)"
                fi
            else
                print_success "DataJoint fully imported: ✓"
                print_info "  - SQLite Imported flag: ✓"
                print_info "  - DataJoint tables populated: ✓"
            fi

            # 2. Verify mount and backup exists
            print_info "Checking backup destination..."
            if ! verify_mount >/dev/null 2>&1; then
                print_error "Backup mount not accessible"
                echo "Cannot verify backup exists. Aborting deletion for safety."
                exit 1
            fi

            if [ ! -d "$dest" ]; then
                if [ "$force_delete" = "false" ]; then
                    print_error "Backup directory not found: $dest"
                    echo "No backup exists. Cannot safely delete."
                    echo "Use --force to override (NOT RECOMMENDED)"
                    exit 1
                else
                    print_error "Backup not found (--force used, proceeding anyway)"
                fi
            else
                print_success "Backup exists: $dest"
            fi

            # 3. Verify backup integrity (file counts + sample sizes)
            if [ "$force_delete" = "false" ] && [ -d "$dest" ]; then
                print_info "Verifying backup integrity..."

                # Quick verification: file counts and sizes
                local source_files=$(cd "$source" && find . -type f | sort)
                local dest_files=$(cd "$dest" && find . -type f | sort)
                local source_count=$(echo "$source_files" | grep -c . || echo "0")
                local dest_count=$(echo "$dest_files" | grep -c . || echo "0")

                # Check if empty
                if [ "$source_count" -eq 0 ]; then
                    print_error "Source directory is empty! Nothing to verify."
                    echo "Cannot safely delete an empty directory."
                    exit 1
                fi

                if [ "$source_count" -ne "$dest_count" ]; then
                    print_error "File count mismatch: source=$source_count, backup=$dest_count"
                    echo "Backup verification failed. Cannot safely delete."
                    echo "Run './backup_manager.sh verify $participant_id $session_date' for details"
                    exit 1
                fi

                # Check a sample of file sizes (10 files for speed)
                local sample_size=10
                local size_ok=true
                local checked=0
                while IFS= read -r file && [ $checked -lt $sample_size ]; do
                    checked=$((checked + 1))
                    local source_size=$(stat -c %s "$source/$file" 2>/dev/null)
                    local dest_size=$(stat -c %s "$dest/$file" 2>/dev/null)
                    if [ "$source_size" != "$dest_size" ]; then
                        size_ok=false
                        print_error "File size mismatch detected: $file"
                        break
                    fi
                done <<< "$source_files"

                if [ "$size_ok" = "false" ]; then
                    echo "Backup verification failed. Cannot safely delete."
                    echo "Run './backup_manager.sh verify $participant_id $session_date' for full verification"
                    exit 1
                fi

                print_success "Backup verified (file count: $source_count, sample sizes match)"
            fi

            # 4. Show deletion summary
            local file_count=$(find "$source" -type f | wc -l)
            local total_size=$(du -sh "$source" | cut -f1)

            echo -e "\n${BOLD}${RED}WARNING: About to DELETE local files${NC}"
            echo -e "${BOLD}════════════════════════════════════════${NC}"
            echo "  Path: $source"
            echo "  Files: $file_count"
            echo "  Size: $total_size"
            echo ""
            echo "  DataJoint: $(check_datajoint_imported "$participant_id" "$session_date" && echo "✓ Fully imported (SQLite + DataJoint tables)" || echo "✗ Not fully imported")"
            echo "  Backup: $([ -d "$dest" ] && echo "✓ Exists at $dest" || echo "✗ Not found")"
            echo -e "${BOLD}════════════════════════════════════════${NC}"

            # 5. Confirmation prompt
            echo -e "\n${YELLOW}This action cannot be undone!${NC}"
            echo -e "${YELLOW}Data will only exist in:${NC}"
            echo -e "  1. DataJoint database tables"
            echo -e "  2. Network backup at: $dest"
            echo ""
            echo -n "Type 'DELETE' (in capital letters) to confirm: "
            read -r confirmation

            if [ "$confirmation" != "DELETE" ]; then
                print_info "Deletion cancelled."
                exit 0
            fi

            # 6. Execute deletion
            echo -e "\n${BOLD}Deleting local files...${NC}"
            if rm -rf "$source"; then
                print_success "Successfully deleted: $source"
            else
                print_error "Failed to delete: $source"
                exit 1
            fi

            # 7. Verify deletion
            if [ -d "$source" ]; then
                print_error "Directory still exists after deletion attempt!"
                exit 1
            fi

            echo -e "\n${GREEN}${BOLD}✓ Deletion completed successfully${NC}"
            echo "  Freed up approximately: $total_size"
            echo "  Data preserved in:"
            echo "    - DataJoint database"
            echo "    - Network backup: $dest"
            ;;

        *)
            print_error "Unknown command: $command"
            echo "Run '$0' without arguments to see usage"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"

exit 0
