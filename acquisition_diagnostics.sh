#!/bin/bash

################################################################################
# MultiCameraTracking Acquisition Diagnostics Script
#
# Purpose: Collect diagnostic information for debugging acquisition issues
# Usage: sudo ./acquisition_diagnostics.sh [-v|--verbose]
#        -v, --verbose: Show detailed output (DHCP leases, config files, etc.)
# Output: Creates /tmp/acquisition_diagnostics_YYYYMMDD_HHMMSS.log
#         and /tmp/acquisition_diagnostics_YYYYMMDD_HHMMSS.tar.gz
#
# This script should be run from the MultiCameraTracking repository root
# on the Ubuntu laptop used for acquisition (HOST system, not inside container)
################################################################################

# Parse command line arguments
VERBOSE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Usage: $0 [-v|--verbose]"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'  # Brighter than blue for better visibility
NC='\033[0m' # No Color
BOLD='\033[1m'

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/tmp/acquisition_diagnostics_${TIMESTAMP}.log"
TARBALL="/tmp/acquisition_diagnostics_${TIMESTAMP}.tar.gz"
TEMP_DIR="/tmp/acquisition_diagnostics_${TIMESTAMP}"

# Error counters
ERROR_COUNT=0
WARNING_COUNT=0

# Create temp directory for collecting files
mkdir -p "${TEMP_DIR}"

################################################################################
# Helper Functions
################################################################################

# Print section header
print_section() {
    local section_name="$1"
    echo "" | tee -a "${LOG_FILE}"
    echo "================================================================================" | tee -a "${LOG_FILE}"
    echo -e "${BOLD}${CYAN}${section_name}${NC}" | tee -a "${LOG_FILE}"
    echo "================================================================================" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
}

# Print error message
print_error() {
    local message="$1"
    echo -e "${RED}[ERROR]${NC} ${message}" | tee -a "${LOG_FILE}"
    ERROR_COUNT=$((ERROR_COUNT + 1))
}

# Print warning message
print_warning() {
    local message="$1"
    echo -e "${YELLOW}[WARNING]${NC} ${message}" | tee -a "${LOG_FILE}"
    WARNING_COUNT=$((WARNING_COUNT + 1))
}

# Print success message
print_success() {
    local message="$1"
    echo -e "${GREEN}[OK]${NC} ${message}" | tee -a "${LOG_FILE}"
}

# Print info message
print_info() {
    local message="$1"
    echo "${message}" | tee -a "${LOG_FILE}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Run command and capture output
run_command() {
    local description="$1"
    local cmd="$2"

    print_info "Running: ${description}"
    echo "$ ${cmd}" >> "${LOG_FILE}"
    eval "${cmd}" 2>&1 | tee -a "${LOG_FILE}"
    echo "" >> "${LOG_FILE}"
}

################################################################################
# Main Diagnostic Functions
################################################################################

collect_system_info() {
    print_section "1. SYSTEM INFORMATION"

    # Get Ubuntu version
    print_info "OS Version:"
    grep "PRETTY_NAME\|VERSION_ID" /etc/os-release | tee -a "${LOG_FILE}"

    # Get kernel version
    print_info ""
    print_info "Kernel Version:"
    uname -r | tee -a "${LOG_FILE}"

    print_info ""
    run_command "Current Timestamp" "date"
}

collect_env_config() {
    print_section "2. ENVIRONMENT CONFIGURATION"

    # Check for .env file
    if [ -f ".env" ]; then
        print_success ".env file found"
        print_info ""
        print_info "Contents of .env:"
        cat .env | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"

        # Don't copy .env to temp directory (not needed in tarball)

        # Load environment variables using a safer method for /bin/sh compatibility
        while IFS='=' read -r key value; do
            # Skip comments and empty lines
            case "$key" in
                ''|\#*) continue ;;
            esac
            # Remove any quotes from value
            value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")
            # Export the variable
            eval "export $key='$value'"
        done < .env

        # Validate required variables
        print_info "Validating environment variables:"

        if [ -n "$NETWORK_INTERFACE" ]; then
            print_success "NETWORK_INTERFACE = $NETWORK_INTERFACE"
        else
            print_error "NETWORK_INTERFACE not set in .env"
        fi

        if [ -n "$DATA_VOLUME" ]; then
            print_success "DATA_VOLUME = $DATA_VOLUME"
        else
            print_error "DATA_VOLUME not set in .env"
        fi

        if [ -n "$CAMERA_CONFIGS" ]; then
            print_success "CAMERA_CONFIGS = $CAMERA_CONFIGS"
        else
            print_error "CAMERA_CONFIGS not set in .env"
        fi

        # Check disk space on data volume
        print_info ""
        if [ -n "${DATA_VOLUME}" ] && [ -d "${DATA_VOLUME}" ]; then
            print_info "Disk space for recording directory (${DATA_VOLUME}):"
            df -h "${DATA_VOLUME}" | tee -a "${LOG_FILE}"

            local disk_avail_gb
            disk_avail_gb=$(df -BG "${DATA_VOLUME}" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//')
            if [ -n "${disk_avail_gb}" ] && [ "${disk_avail_gb}" -lt 10 ]; then
                print_warning "Low disk space: less than 10GB available"
            else
                print_success "Sufficient disk space available"
            fi
        elif [ -n "${DATA_VOLUME}" ]; then
            print_warning "DATA_VOLUME directory does not exist: ${DATA_VOLUME}"
        fi

    else
        print_error ".env file not found in current directory"
        print_info "Current directory: $(pwd)"
        print_info "Make sure you're running this from the MultiCameraTracking root"
        return 1
    fi
}

collect_network_info() {
    print_section "3. NETWORK CONFIGURATION"

    # NETWORK_INTERFACE is already validated in section 2
    if [ -z "${NETWORK_INTERFACE}" ]; then
        print_error "NETWORK_INTERFACE not available (should have been set in section 2)"
        return 1
    fi

    print_info "Checking interface: ${NETWORK_INTERFACE}"
    print_info ""

    # Check if interface exists
    if ! ip link show "${NETWORK_INTERFACE}" >/dev/null 2>&1; then
        print_error "Interface ${NETWORK_INTERFACE} does not exist or is not connected"
        print_info ""
        print_info "Available interfaces:"
        ip link show | grep -E '^[0-9]+:' | awk '{print $2}' | sed 's/:$//' | tee -a "${LOG_FILE}"
        return 1
    fi

    print_success "Interface ${NETWORK_INTERFACE} exists"

    # Get interface details
    print_info ""
    ip a show "${NETWORK_INTERFACE}" | tee -a "${LOG_FILE}"

    # Check MTU
    print_info ""
    local mtu=$(ip link show "${NETWORK_INTERFACE}" | grep -oP 'mtu \K\d+')
    if [ "${mtu}" = "9000" ]; then
        print_success "MTU is set to 9000 (jumbo frames enabled)"
    else
        print_error "MTU is ${mtu}, should be 9000 for optimal performance"
        print_info "Fix with: sudo sh set_mtu.sh"
    fi

    # Check if interface is up
    if ip link show "${NETWORK_INTERFACE}" | grep -q "state UP"; then
        print_success "Interface ${NETWORK_INTERFACE} is UP"
    else
        print_error "Interface ${NETWORK_INTERFACE} is DOWN"
    fi

    # Check IP address
    local ip_addr=$(ip addr show "${NETWORK_INTERFACE}" | grep -oP 'inet \K[\d.]+')
    if [ -n "${ip_addr}" ]; then
        if [ "${ip_addr}" = "192.168.1.1" ]; then
            print_success "IP address is ${ip_addr} (correct for DHCP server)"
        else
            print_warning "IP address is ${ip_addr}, expected 192.168.1.1 for camera acquisition"
        fi
    else
        print_error "No IP address assigned to ${NETWORK_INTERFACE}"
    fi

    # Check NetworkManager connection
    if command_exists nmcli; then
        print_info ""
        print_info "NetworkManager status:"
        # Check if DHCP-Server profile exists and is active
        if nmcli con show | grep -q "DHCP-Server"; then
            if nmcli con show --active | grep -q "DHCP-Server"; then
                print_success "DHCP-Server profile is ACTIVE"
            else
                print_warning "DHCP-Server profile exists but is NOT ACTIVE"
                print_info "Activate with: nmcli con up id 'DHCP-Server'"
            fi
        else
            print_warning "DHCP-Server profile not found in NetworkManager"
        fi
    fi

    # Check network buffer settings
    print_info ""
    local rmem_max=$(sysctl -n net.core.rmem_max 2>/dev/null)
    if [ -n "${rmem_max}" ] && [ "${rmem_max}" -ge 10000000 ]; then
        print_success "net.core.rmem_max is set to ${rmem_max} (>= 10000000)"
    else
        print_warning "net.core.rmem_max is ${rmem_max}, recommended >= 10000000"
        print_info "This is set by the set_mtu.sh script"
    fi
}

collect_dhcp_info() {
    print_section "4. DHCP SERVER STATUS"

    # Check DHCP server status
    print_info "Checking isc-dhcp-server status..."
    if systemctl is-active --quiet isc-dhcp-server 2>/dev/null; then
        print_success "isc-dhcp-server is running"
    elif service isc-dhcp-server status >/dev/null 2>&1; then
        print_success "isc-dhcp-server is running"
    else
        print_error "isc-dhcp-server is NOT running"
        print_info "Start with: sudo service isc-dhcp-server start"
    fi

    run_command "DHCP Server Detailed Status" "sudo service isc-dhcp-server status"

    # Check for recent DHCP logs
    print_info ""
    print_info "Recent DHCP server logs (last 50 lines):"
    if command_exists journalctl; then
        sudo journalctl -u isc-dhcp-server -n 50 --no-pager 2>&1 | tee -a "${LOG_FILE}"
    else
        print_warning "journalctl not available"
    fi

    # Check DHCP leases (verbose mode or always to log file)
    print_info "" >> "${LOG_FILE}"
    if [ -f "/var/lib/dhcp/dhcpd.leases" ]; then
        echo "DHCP Leases (active camera IPs):" >> "${LOG_FILE}"
        sudo grep -E "lease|hardware ethernet|client-hostname" /var/lib/dhcp/dhcpd.leases 2>/dev/null | tail -30 >> "${LOG_FILE}"

        if [ "$VERBOSE" = true ]; then
            print_info ""
            print_info "DHCP Leases (active camera IPs):"
            sudo grep -E "lease|hardware ethernet|client-hostname" /var/lib/dhcp/dhcpd.leases 2>/dev/null | tail -30
        fi
    else
        echo "DHCP leases file not found at /var/lib/dhcp/dhcpd.leases" >> "${LOG_FILE}"
    fi
}

collect_camera_info() {
    print_section "5. CAMERA DETECTION"

    # Run camera detection using Docker Compose
    # Use the 'test' service which has proper network_mode: host and volumes mounted
    print_info "Running camera detection script via Docker..."
    print_info "(This may take 10-15 seconds)"
    print_info ""

    # Save output to temp file to avoid command substitution issues
    local camera_output_file="${TEMP_DIR}/camera_check.txt"

    # Run camera check using docker compose (similar to make test/reset)
    # Need to override the default command from the test service
    docker compose run --rm --entrypoint python3 test /Mocap/multi_camera/acquisition/diagnostics/check_active_cameras.py > "${camera_output_file}" 2>&1

    # Display and log the output
    cat "${camera_output_file}" | tee -a "${LOG_FILE}"

    print_info ""

    # Parse the output to check for issues
    if grep -q "Serial:" "${camera_output_file}"; then
        local num_cameras
        num_cameras=$(grep -c "Serial:" "${camera_output_file}")
        print_success "Detected ${num_cameras} camera(s) on network"

        # Check if any are in use
        if grep -q "In Use" "${camera_output_file}"; then
            print_warning "Some cameras are currently in use"
        fi

        # Check if any are available
        if grep -q "Available" "${camera_output_file}"; then
            print_success "Some cameras are available for use"
        fi
    else
        print_error "No cameras detected on network"
        print_info ""
        print_info "Troubleshooting steps:"
        print_info "  1. Check cameras are powered (via PoE from switch)"
        print_info "  2. Look for double-blinking green LED on camera back"
        print_info "  3. Verify DHCP server is running (see section 4)"
        print_info "  4. Verify MTU is set to 9000 (see section 3)"
        print_info "  5. Check network cables are connected properly"
        print_info "  6. Try: sudo service isc-dhcp-server restart"
    fi

    # Check for camera config files
    print_info ""
    if [ -n "${CAMERA_CONFIGS}" ] && [ -d "${CAMERA_CONFIGS}" ]; then
        print_success "Camera configs directory found: ${CAMERA_CONFIGS}"

        # Always log to file
        echo "Available config files:" >> "${LOG_FILE}"
        ls -lh "${CAMERA_CONFIGS}"/*.yaml 2>/dev/null >> "${LOG_FILE}"

        # Show in console if verbose
        if [ "$VERBOSE" = true ]; then
            print_info "Available config files:"
            ls -lh "${CAMERA_CONFIGS}"/*.yaml 2>/dev/null
        fi

        if [ ! -f "${CAMERA_CONFIGS}"/*.yaml 2>/dev/null ]; then
            print_warning "No .yaml config files found in ${CAMERA_CONFIGS}"
        fi
    else
        print_warning "Camera configs directory not found: ${CAMERA_CONFIGS}"
    fi
}

collect_logs() {
    print_section "6. SYSTEM LOGS"

    # Save full logs to temp directory for tarball
    print_info "Saving full system logs to tarball..."
    dmesg 2>/dev/null > "${TEMP_DIR}/dmesg_full.log"
    if command_exists journalctl; then
        sudo journalctl --no-pager 2>/dev/null > "${TEMP_DIR}/journalctl_full.log"
    fi

    print_info "Last 200 lines of dmesg (filtered for network/USB/camera/error):"
    dmesg 2>/dev/null | tail -200 | grep -iE 'network|eth|usb|camera|gige|flir|error|fail|warn' | tee -a "${LOG_FILE}"

    if [ $? -ne 0 ]; then
        print_warning "Unable to read dmesg (may need sudo)"
    fi

    print_info ""
    print_info "Last 200 lines of journalctl (filtered for relevant services):"
    if command_exists journalctl; then
        sudo journalctl -n 200 --no-pager 2>/dev/null | grep -iE 'dhcp|network|docker|error|fail' | tee -a "${LOG_FILE}"
    else
        print_warning "journalctl not available"
    fi
}

check_recent_recordings() {
    # Only show section header if verbose
    if [ "$VERBOSE" = true ]; then
        print_section "7. RECENT RECORDINGS CHECK"
    else
        echo "" >> "${LOG_FILE}"
        echo "================================================================================" >> "${LOG_FILE}"
        echo "7. RECENT RECORDINGS CHECK" >> "${LOG_FILE}"
        echo "================================================================================" >> "${LOG_FILE}"
        echo "" >> "${LOG_FILE}"
    fi

    # Check data directory from .env
    if [ -z "${DATA_VOLUME}" ]; then
        echo "DATA_VOLUME not set in .env" >> "${LOG_FILE}"
        return 1
    fi

    if [ ! -d "${DATA_VOLUME}" ]; then
        echo "Data volume not accessible: ${DATA_VOLUME}" >> "${LOG_FILE}"
        return 1
    fi

    echo "Data volume found: ${DATA_VOLUME}" >> "${LOG_FILE}"
    if [ "$VERBOSE" = true ]; then
        print_success "Data volume found: ${DATA_VOLUME}"
    fi

    # Find most recent directory with recordings (look for directories with timestamps)
    local recent_dir
    recent_dir=$(find "${DATA_VOLUME}" -maxdepth 3 -type f -name "*.mp4" -printf '%T@ %h\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)

    if [ -n "${recent_dir}" ]; then
        echo "Most recent recording directory: ${recent_dir}" >> "${LOG_FILE}"
        if [ "$VERBOSE" = true ]; then
            print_info "Most recent recording directory: ${recent_dir}"
        fi

        # List video files (always to log)
        echo "" >> "${LOG_FILE}"
        echo "Video files in recent directory:" >> "${LOG_FILE}"
        ls -lh "${recent_dir}"/*.mp4 2>/dev/null >> "${LOG_FILE}"
        local num_videos
        num_videos=$(ls -1 "${recent_dir}"/*.mp4 2>/dev/null | wc -l)

        if [ "$VERBOSE" = true ]; then
            print_info ""
            print_info "Video files in recent directory:"
            ls -lh "${recent_dir}"/*.mp4 2>/dev/null
        fi

        if [ "${num_videos}" -gt 0 ]; then
            echo "Found ${num_videos} video file(s)" >> "${LOG_FILE}"
            if [ "$VERBOSE" = true ]; then
                print_success "Found ${num_videos} video file(s)"
            fi

            # Check file sizes - warn if any are suspiciously small
            local small_videos
            small_videos=$(find "${recent_dir}" -name "*.mp4" -size -1M 2>/dev/null)
            if [ -n "${small_videos}" ]; then
                echo "WARNING: Some video files are smaller than 1MB (may be incomplete):" >> "${LOG_FILE}"
                echo "${small_videos}" >> "${LOG_FILE}"
                if [ "$VERBOSE" = true ]; then
                    print_warning "Some video files are smaller than 1MB (may be incomplete):"
                    echo "${small_videos}"
                fi
            fi
        else
            echo "No video files found in recent directory" >> "${LOG_FILE}"
        fi

        # Check JSON files
        echo "" >> "${LOG_FILE}"
        echo "Metadata files in recent directory:" >> "${LOG_FILE}"
        ls -lh "${recent_dir}"/*.json 2>/dev/null >> "${LOG_FILE}"
        local json_file
        json_file=$(ls "${recent_dir}"/*.json 2>/dev/null | head -1)

        if [ "$VERBOSE" = true ]; then
            print_info ""
            print_info "Metadata files in recent directory:"
            ls -lh "${recent_dir}"/*.json 2>/dev/null
        fi

        if [ -n "${json_file}" ]; then
            echo "Found metadata JSON file: $(basename ${json_file})" >> "${LOG_FILE}"
            if [ "$VERBOSE" = true ]; then
                print_success "Found metadata JSON file: $(basename ${json_file})"
            fi

            # Analyze JSON file using Docker
            echo "" >> "${LOG_FILE}"
            echo "Analyzing JSON metadata for synchronization quality..." >> "${LOG_FILE}"
            if [ "$VERBOSE" = true ]; then
                print_info ""
                print_info "Analyzing JSON metadata for synchronization quality..."
            fi

            # Convert host path to container path for Docker
            local json_file_container
            json_file_container=$(echo "${json_file}" | sed "s|${DATA_VOLUME}|/data|")

            # Save analysis to temp file
            local json_analysis_file="${TEMP_DIR}/json_analysis.txt"

            # Run Python analysis in Docker container with proper entrypoint override
            docker compose run --rm --entrypoint python3 test -c "
import json
import numpy as np
import sys

json_file = '${json_file}'

try:
    with open(json_file, 'r') as f:
        data = json.load(f)

    timestamps = np.array(data['timestamps'])
    dt = (timestamps - timestamps[0, 0]) / 1e9
    spread = np.max(dt, axis=1) - np.min(dt, axis=1)
    max_spread_ms = np.max(spread) * 1000

    print(f'Filename: {json_file}')
    print(f'Camera serials: {data[\"serials\"]}')
    print(f'Number of frames recorded: {len(timestamps)}')
    print(f'Max timestamp spread: {max_spread_ms:.3f} ms')

    if max_spread_ms > 1.0:
        print(f'WARNING: Timestamp spread exceeds 1ms threshold')
    else:
        print(f'GOOD: Timestamp synchronization is within acceptable limits')

except FileNotFoundError:
    print(f'Error: Could not find JSON file at {json_file}')
except Exception as e:
    print(f'Error parsing JSON: {e}')
" > "${json_analysis_file}" 2>&1

            # Always log the output
            cat "${json_analysis_file}" >> "${LOG_FILE}"

            # Display if verbose
            if [ "$VERBOSE" = true ]; then
                cat "${json_analysis_file}"
            fi

            # Check if sync was good (always show this summary)
            if grep -q "GOOD:" "${json_analysis_file}"; then
                echo "Timestamp synchronization looks good" >> "${LOG_FILE}"
            elif grep -q "WARNING:" "${json_analysis_file}"; then
                echo "WARNING: Timestamp synchronization issues detected" >> "${LOG_FILE}"
            elif grep -q "Error" "${json_analysis_file}"; then
                echo "WARNING: Could not analyze JSON file" >> "${LOG_FILE}"
            fi
        else
            echo "No metadata JSON files found in recent directory" >> "${LOG_FILE}"
        fi
    else
        echo "No recent recording directories found in ${DATA_VOLUME}" >> "${LOG_FILE}"
    fi
}

check_hardware() {
    print_section "8. HARDWARE INFORMATION"

    print_info "PCI Devices (Ethernet/Thunderbolt adapters):"
    lspci 2>/dev/null | grep -iE 'ethernet|network|thunderbolt' | tee -a "${LOG_FILE}"

    # Check ethernet adapter details if ethtool is available (verbose mode or log only)
    if command_exists ethtool && [ -n "${NETWORK_INTERFACE}" ]; then
        echo "" >> "${LOG_FILE}"
        echo "Ethernet Adapter Details for ${NETWORK_INTERFACE}:" >> "${LOG_FILE}"
        sudo ethtool "${NETWORK_INTERFACE}" 2>&1 >> "${LOG_FILE}"

        if [ "$VERBOSE" = true ]; then
            print_info ""
            run_command "Ethernet Adapter Details for ${NETWORK_INTERFACE}" "sudo ethtool ${NETWORK_INTERFACE}"
        fi
    fi
}

generate_summary() {
    print_section "9. DIAGNOSTIC SUMMARY"

    print_info "Diagnostic run completed at: $(date)"
    print_info ""

    # Provide summary of key findings
    print_info "=== Key Configuration Checks ==="
    echo "" | tee -a "${LOG_FILE}"

    # Network interface check
    if [ -n "${NETWORK_INTERFACE}" ] && ip link show "${NETWORK_INTERFACE}" >/dev/null 2>&1; then
        local mtu=$(ip link show "${NETWORK_INTERFACE}" 2>/dev/null | grep -oP 'mtu \K\d+')
        if [ "${mtu}" = "9000" ]; then
            echo -e "${GREEN}✓${NC} Network interface ${NETWORK_INTERFACE} with MTU 9000" | tee -a "${LOG_FILE}"
        else
            echo -e "${RED}✗${NC} Network interface MTU is ${mtu} (should be 9000)" | tee -a "${LOG_FILE}"
        fi
    else
        echo -e "${RED}✗${NC} Network interface ${NETWORK_INTERFACE} not found" | tee -a "${LOG_FILE}"
    fi

    # DHCP check
    if systemctl is-active --quiet isc-dhcp-server 2>/dev/null || service isc-dhcp-server status >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} DHCP server is running" | tee -a "${LOG_FILE}"
    else
        echo -e "${RED}✗${NC} DHCP server is not running" | tee -a "${LOG_FILE}"
    fi

    echo "" | tee -a "${LOG_FILE}"
    print_info "=== Issue Summary ==="
    echo "" | tee -a "${LOG_FILE}"

    if [ ${ERROR_COUNT} -eq 0 ] && [ ${WARNING_COUNT} -eq 0 ]; then
        print_success "No errors or warnings detected!"
        print_info "System appears to be configured correctly for acquisition."
    else
        if [ ${ERROR_COUNT} -gt 0 ]; then
            print_error "Found ${ERROR_COUNT} error(s) - these must be fixed"
        fi
        if [ ${WARNING_COUNT} -gt 0 ]; then
            print_warning "Found ${WARNING_COUNT} warning(s) - review recommended"
        fi
        print_info ""
        print_info "Review the sections above for detailed information."
    fi

    print_info ""
    print_info "=== Output Files ==="
    print_info "Log file: ${LOG_FILE}"
    print_info "Tarball:  ${TARBALL}"
    print_info "  (contains: diagnostic log, full dmesg, full journalctl)"
}

create_tarball() {
    print_info "Creating tarball with diagnostic data..."

    # Copy log file to temp directory
    cp "${LOG_FILE}" "${TEMP_DIR}/"

    # Create tarball
    tar -czf "${TARBALL}" -C /tmp "$(basename ${TEMP_DIR})" 2>/dev/null

    if [ $? -eq 0 ]; then
        print_success "Tarball created: ${TARBALL}"
        local tarball_size=$(du -h "${TARBALL}" | cut -f1)
        print_info "Size: ${tarball_size}"
    else
        print_error "Failed to create tarball"
    fi

    # Cleanup temp directory
    rm -rf "${TEMP_DIR}"
}

################################################################################
# Main Execution
################################################################################

main() {
    # Print header
    clear
    echo "================================================================================"
    echo -e "${BOLD}MultiCameraTracking Acquisition Diagnostics${NC}"
    echo "================================================================================"
    echo ""
    echo "Starting diagnostic collection at $(date)"
    echo "Log file: ${LOG_FILE}"
    echo ""

    # Check if we're in the right directory
    if [ ! -f "docker-compose.yml" ]; then
        echo -e "${RED}ERROR:${NC} docker-compose.yml not found"
        echo "Please run this script from the MultiCameraTracking repository root"
        exit 1
    fi

    # Initialize log file
    echo "MultiCameraTracking Acquisition Diagnostics" > "${LOG_FILE}"
    echo "Generated: $(date)" >> "${LOG_FILE}"
    echo "Hostname: $(hostname)" >> "${LOG_FILE}"
    echo "================================================================================" >> "${LOG_FILE}"
    echo "" >> "${LOG_FILE}"

    # Check if running as root
    if [ "$(id -u)" -ne 0 ]; then
        print_warning "Not running as root. Some checks may require sudo."
        print_info "For complete diagnostics, run with: sudo ./acquisition_diagnostics.sh"
        echo ""
    fi

    # Run all diagnostic functions
    collect_system_info
    collect_env_config
    collect_network_info
    collect_dhcp_info
    collect_camera_info
    collect_logs
    check_recent_recordings
    check_hardware

    # Create tarball
    create_tarball

    # Generate summary
    generate_summary
}

# Run main function
main

exit 0
