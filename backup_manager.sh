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

# Run Python backup script in Docker container
run_backup_script() {
    local script_args="$@"

    # Run using docker compose with the backup service
    # Use --rm to clean up container after execution
    docker compose run --rm --entrypoint python3 backup /Mocap/scripts/backup_sync.py ${script_args}
}

# Main command dispatcher
main() {
    check_directory

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
            local participant_id="$1"
            local session_date="$2"
            shift 2
            run_backup_script --session "${participant_id}" "${session_date}" "$@"
            ;;

        batch)
            if [ $# -lt 1 ]; then
                print_error "Usage: $0 batch <start_date> [end_date] [--dry-run]"
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
            if [ $# -lt 1 ]; then
                print_error "Usage: $0 status <participant_id> <session_date> OR $0 status <session_date>"
                exit 1
            fi

            # Check if it's a single date (8 digits) or participant+date (2 args)
            if [ $# -eq 1 ] && [[ "$1" =~ ^[0-9]{8}$ ]]; then
                # Single date - show all sessions for that date
                run_backup_script --status "$1"
            else
                # Participant and date
                run_backup_script --status "$@"
            fi
            ;;

        status-range)
            # Allow 0, 1, or 2 date arguments
            run_backup_script --status-range "$@"
            ;;

        verify)
            if [ $# -ne 2 ]; then
                print_error "Usage: $0 verify <participant_id> <session_date>"
                exit 1
            fi
            run_backup_script --verify "$1" "$2"
            ;;

        delete)
            if [ $# -lt 2 ]; then
                print_error "Usage: $0 delete <participant_id> <session_date> [--force]"
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
