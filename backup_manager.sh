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

    if [ ! -d "$source" ]; then
        print_error "Source path not found: $source"
        return 1
    fi

    local cmd="rsync $RSYNC_FLAGS"
    [ "$dry_run" = "true" ] && cmd="$cmd --dry-run"
    cmd="$cmd \"${source}/\" \"${dest}/\""

    print_info "Running: $cmd"
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
        echo "  $0 status-range [start_date] [end_date]"
        echo "      Show status for sessions in date range (no args = all sessions)"
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
        echo "  $0 status-range 20250101"
        echo "  $0 status-range 20250101 20250131"
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
                    # TODO: Add verification logic here
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
            local extra_args=""

            # Check if next arg is a date or a flag
            if [ $# -gt 0 ] && [[ "$1" =~ ^[0-9]{8}$ ]]; then
                end_date="$1"
                shift
                extra_args="--end-date ${end_date}"
            fi

            run_backup_script --start-date "${start_date}" ${extra_args} "$@"
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

            if [ -d "$source" ]; then
                local source_count=$(find "$source" -type f | wc -l)
                local source_size=$(du -sh "$source" | cut -f1)
                print_success "Source exists: $source ($source_count files, $source_size)"
            else
                print_error "Source not found: $source"
            fi

            if verify_mount >/dev/null 2>&1 && [ -d "$dest" ]; then
                local dest_count=$(find "$dest" -type f | wc -l)
                local dest_size=$(du -sh "$dest" | cut -f1)
                print_success "Backup exists: $dest ($dest_count files, $dest_size)"

                # Check if counts match
                if [ "$source_count" -eq "$dest_count" ]; then
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
            echo -e "\n${BOLD}Querying sessions from database...${NC}\n"

            local start_date="${1:-}"
            local end_date="${2:-}"

            printf "%-12s %-15s %-4s %-8s %-10s %-6s\n" "Date" "Participant" "DJ" "Backup" "Verified" "Safe"
            echo "================================================================================"

            while IFS=$'\t' read -r participant_id session_date dj_imported; do
                local dest=$(get_dest_path "$participant_id" "$session_date")

                # Check if backup exists
                local backup_exists="✗"
                local verified="✗"
                if verify_mount >/dev/null 2>&1 && [ -d "$dest" ]; then
                    backup_exists="✓"

                    # Quick verification - just check file counts
                    local source=$(get_source_path "$participant_id" "$session_date")
                    if [ -d "$source" ]; then
                        local source_count=$(find "$source" -type f 2>/dev/null | wc -l)
                        local dest_count=$(find "$dest" -type f 2>/dev/null | wc -l)
                        [ "$source_count" -eq "$dest_count" ] && verified="✓"
                    fi
                fi

                # Safe to delete?
                local safe="✗"
                [ "$dj_imported" = "True" ] && [ "$backup_exists" = "✓" ] && [ "$verified" = "✓" ] && safe="✓"

                local dj_mark=$([ "$dj_imported" = "True" ] && echo "✓" || echo "✗")

                printf "%-12s %-15s %-4s %-8s %-10s %-6s\n" \
                    "$session_date" "$participant_id" "$dj_mark" "$backup_exists" "$verified" "$safe"

            done < <(get_sessions_from_db "$start_date" "$end_date")
            ;;

        verify)
            if [ $# -ne 2 ]; then
                print_error "Usage: $0 verify <participant_id> <session_date>"
                exit 1
            fi

            # Verify mount before verification
            if ! verify_mount; then
                exit 1
            fi

            run_backup_script --verify "$1" "$2"
            ;;

        delete)
            if [ $# -lt 2 ]; then
                print_error "Usage: $0 delete <participant_id> <session_date> [--force]"
                exit 1
            fi

            # Verify mount before delete (need to ensure backup exists)
            if ! verify_mount; then
                exit 1
            fi

            local participant_id="$1"
            local session_date="$2"
            shift 2

            # Need -it flag for interactive confirmation prompt
            docker compose run --rm -it --entrypoint python3 backup /Mocap/scripts/backup_sync.py --safe-delete "${participant_id}" "${session_date}" "$@"
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
