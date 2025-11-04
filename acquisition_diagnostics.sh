#!/bin/bash

################################################################################
# MultiCameraTracking Acquisition Diagnostics Script
#
# Purpose: Collect diagnostic information for debugging acquisition issues
# Usage: sudo ./acquisition_diagnostics.sh
# Output: Creates /tmp/acquisition_diagnostics_YYYYMMDD_HHMMSS.log
#         and /tmp/acquisition_diagnostics_YYYYMMDD_HHMMSS.tar.gz
#
# This script should be run from the MultiCameraTracking repository root
# on the Ubuntu laptop used for acquisition (HOST system, not inside container)
################################################################################

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
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
    echo -e "${BOLD}${BLUE}${section_name}${NC}" | tee -a "${LOG_FILE}"
    echo "================================================================================" | tee -a "${LOG_FILE}"
    echo "" | tee -a "${LOG_FILE}"
}

# Print error message
print_error() {
    local message="$1"
    echo -e "${RED}[ERROR]${NC} ${message}" | tee -a "${LOG_FILE}"
    ((ERROR_COUNT++))
}

# Print warning message
print_warning() {
    local message="$1"
    echo -e "${YELLOW}[WARNING]${NC} ${message}" | tee -a "${LOG_FILE}"
    ((WARNING_COUNT++))
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

    run_command "OS Version" "cat /etc/os-release"
    run_command "Kernel Version" "uname -a"
    run_command "Timestamp" "date"
    run_command "Uptime and Load Average" "uptime"
    run_command "CPU Info" "lscpu | grep -E 'Model name|CPU\(s\)|Thread'"
    run_command "Memory Usage" "free -h"
    run_command "Disk Space" "df -h"

    # Check disk space on data volume if accessible
    if [ -n "${DATA_VOLUME}" ] && [ -d "${DATA_VOLUME}" ]; then
        local disk_avail=$(df -h "${DATA_VOLUME}" | awk 'NR==2 {print $4}')
        local disk_avail_gb=$(df -BG "${DATA_VOLUME}" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//')

        if [ -n "${disk_avail_gb}" ] && [ "${disk_avail_gb}" -lt 10 ]; then
            print_warning "Low disk space on ${DATA_VOLUME}: ${disk_avail} available"
        elif [ -n "${disk_avail}" ]; then
            print_success "Disk space on ${DATA_VOLUME}: ${disk_avail} available"
        fi
    fi
}

collect_env_config() {
    print_section "2. ENVIRONMENT CONFIGURATION"

    # Check for .env file
    if [ -f ".env" ]; then
        print_success ".env file found"
        print_info "Contents of .env:"
        cat .env | tee -a "${LOG_FILE}"
        echo "" | tee -a "${LOG_FILE}"

        # Copy .env to temp directory
        cp .env "${TEMP_DIR}/.env"

        # Source the .env file to get variables
        set -a
        source .env
        set +a

        export NETWORK_INTERFACE DATA_VOLUME CAMERA_CONFIGS
    else
        print_error ".env file not found in current directory"
        print_info "Current directory: $(pwd)"
        print_info "Make sure you're running this from the MultiCameraTracking root"
        return 1
    fi
}

collect_network_info() {
    print_section "3. NETWORK CONFIGURATION"

    if [ -z "${NETWORK_INTERFACE}" ]; then
        print_error "NETWORK_INTERFACE not set in .env file"
        print_info "Showing all network interfaces:"
        run_command "All Network Interfaces" "ip a"
    else
        print_success "NETWORK_INTERFACE set to: ${NETWORK_INTERFACE}"

        # Check if interface exists
        if ip link show "${NETWORK_INTERFACE}" >/dev/null 2>&1; then
            print_success "Interface ${NETWORK_INTERFACE} exists"

            # Get interface details
            run_command "Interface ${NETWORK_INTERFACE} Details" "ip a show ${NETWORK_INTERFACE}"

            # Check MTU
            local mtu=$(ip link show "${NETWORK_INTERFACE}" | grep -oP 'mtu \K\d+')
            if [ "${mtu}" = "9000" ]; then
                print_success "MTU is set to 9000 (jumbo frames enabled)"
            else
                print_error "MTU is ${mtu}, should be 9000 for optimal performance"
                print_info "Run: sudo sh set_mtu.sh"
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
                print_success "IP address: ${ip_addr}"
                if [ "${ip_addr}" = "192.168.1.1" ]; then
                    print_success "IP address matches expected DHCP server address"
                else
                    print_warning "IP address is ${ip_addr}, expected 192.168.1.1"
                fi
            else
                print_error "No IP address assigned to ${NETWORK_INTERFACE}"
            fi

        else
            print_error "Interface ${NETWORK_INTERFACE} does not exist or is not connected"
            print_info "Available interfaces:"
            ip link show | grep -E '^[0-9]+:' | tee -a "${LOG_FILE}"
        fi
    fi

    # Check NetworkManager connection
    if command_exists nmcli; then
        print_info ""
        run_command "NetworkManager Connections" "nmcli con show"

        # Check if DHCP-Server profile exists and is active
        if nmcli con show | grep -q "DHCP-Server"; then
            print_success "DHCP-Server profile exists"
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
    run_command "Network Buffer Settings" "sysctl net.core.rmem_max net.core.rmem_default"

    local rmem_max=$(sysctl -n net.core.rmem_max 2>/dev/null)
    if [ -n "${rmem_max}" ] && [ "${rmem_max}" -ge 10000000 ]; then
        print_success "net.core.rmem_max is set to ${rmem_max} (>= 10000000)"
    else
        print_warning "net.core.rmem_max is ${rmem_max}, recommended >= 10000000"
        print_info "This is set by set_mtu.sh script"
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

    # Check DHCP leases
    print_info ""
    if [ -f "/var/lib/dhcp/dhcpd.leases" ]; then
        print_info "DHCP Leases (active camera IPs):"
        sudo grep -E "lease|hardware ethernet|client-hostname" /var/lib/dhcp/dhcpd.leases 2>/dev/null | tail -30 | tee -a "${LOG_FILE}"
    else
        print_warning "DHCP leases file not found at /var/lib/dhcp/dhcpd.leases"
    fi
}

collect_camera_info() {
    print_section "5. CAMERA DETECTION"

    # Check if Docker is available
    if ! command_exists docker; then
        print_error "Docker not found. Cannot check cameras."
        return 1
    fi

    # Check if Docker image exists
    if ! docker image inspect peabody124/mocap >/dev/null 2>&1; then
        print_error "Docker image peabody124/mocap not found"
        print_info "Build with: make build-mocap"
        return 1
    else
        print_success "Docker image peabody124/mocap found"
    fi

    # Run camera detection using Docker
    print_info "Running camera detection script via Docker..."
    print_info "(This may take 10-15 seconds)"

    # Create a temporary Docker Compose service for diagnostics
    local camera_check_output=$(docker compose run --rm -T mocap python3 /Mocap/multi_camera/acquisition/diagnostics/check_active_cameras.py 2>&1)

    echo "${camera_check_output}" | tee -a "${LOG_FILE}"

    # Parse the output to check for issues
    if echo "${camera_check_output}" | grep -q "Serial:"; then
        local num_cameras=$(echo "${camera_check_output}" | grep -c "Serial:")
        print_success "Detected ${num_cameras} camera(s) on network"

        # Check if any are in use
        if echo "${camera_check_output}" | grep -q "In Use"; then
            print_warning "Some cameras are currently in use"
        fi

        # Check if any are available
        if echo "${camera_check_output}" | grep -q "Available"; then
            print_success "Some cameras are available for use"
        fi
    else
        print_error "No cameras detected on network"
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
        print_info "Available config files:"
        ls -lh "${CAMERA_CONFIGS}"/*.yaml 2>/dev/null | tee -a "${LOG_FILE}"

        if [ $? -ne 0 ]; then
            print_warning "No .yaml config files found in ${CAMERA_CONFIGS}"
        fi

        # Copy config files to temp directory
        cp "${CAMERA_CONFIGS}"/*.yaml "${TEMP_DIR}/" 2>/dev/null
    else
        print_warning "Camera configs directory not found: ${CAMERA_CONFIGS}"
    fi
}

collect_logs() {
    print_section "6. SYSTEM LOGS"

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
    print_section "7. RECENT RECORDINGS CHECK"

    # Check data directory from .env
    if [ -z "${DATA_VOLUME}" ]; then
        print_warning "DATA_VOLUME not set in .env"
        return 1
    fi

    if [ ! -d "${DATA_VOLUME}" ]; then
        print_error "Data volume not accessible: ${DATA_VOLUME}"
        return 1
    fi

    print_success "Data volume found: ${DATA_VOLUME}"

    # Find most recent directory with recordings (look for directories with timestamps)
    local recent_dir=$(find "${DATA_VOLUME}" -maxdepth 3 -type f -name "*.mp4" -printf '%T@ %h\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2)

    if [ -n "${recent_dir}" ]; then
        print_info "Most recent recording directory: ${recent_dir}"

        # List video files
        print_info ""
        print_info "Video files in recent directory:"
        ls -lh "${recent_dir}"/*.mp4 2>/dev/null | tee -a "${LOG_FILE}"
        local num_videos=$(ls -1 "${recent_dir}"/*.mp4 2>/dev/null | wc -l)

        if [ "${num_videos}" -gt 0 ]; then
            print_success "Found ${num_videos} video file(s)"

            # Check file sizes - warn if any are suspiciously small
            local small_videos=$(find "${recent_dir}" -name "*.mp4" -size -1M 2>/dev/null)
            if [ -n "${small_videos}" ]; then
                print_warning "Some video files are smaller than 1MB (may be incomplete):"
                echo "${small_videos}" | tee -a "${LOG_FILE}"
            fi
        else
            print_warning "No video files found in recent directory"
        fi

        # Check JSON files
        print_info ""
        print_info "Metadata files in recent directory:"
        ls -lh "${recent_dir}"/*.json 2>/dev/null | tee -a "${LOG_FILE}"
        local json_file=$(ls "${recent_dir}"/*.json 2>/dev/null | head -1)

        if [ -n "${json_file}" ]; then
            print_success "Found metadata JSON file: $(basename ${json_file})"

            # Analyze JSON file using Docker
            print_info ""
            print_info "Analyzing JSON metadata for synchronization quality..."

            # Create a simple Python script to analyze the JSON without matplotlib
            local analysis_output=$(docker compose run --rm -T mocap python3 -c "
import json
import numpy as np
import sys

json_file = '${json_file}'
# Convert host path to container path
json_file = json_file.replace('${DATA_VOLUME}', '/data')

try:
    with open(json_file, 'r') as f:
        data = json.load(f)

    timestamps = np.array(data['timestamps'])
    dt = (timestamps - timestamps[0, 0]) / 1e9
    spread = np.max(dt, axis=1) - np.min(dt, axis=1)
    max_spread_ms = np.max(spread) * 1000

    print(f'Camera serials: {data[\"serials\"]}')
    print(f'Number of frames recorded: {len(timestamps)}')
    print(f'Max timestamp spread: {max_spread_ms:.3f} ms')

    if max_spread_ms > 1.0:
        print(f'WARNING: Timestamp spread exceeds 1ms threshold')
        sys.exit(1)
    else:
        print(f'GOOD: Timestamp synchronization is within acceptable limits')
        sys.exit(0)

except FileNotFoundError:
    print(f'Error: Could not find JSON file')
    sys.exit(2)
except Exception as e:
    print(f'Error parsing JSON: {e}')
    sys.exit(2)
" 2>&1)

            echo "${analysis_output}" | tee -a "${LOG_FILE}"

            # Check exit status to determine if sync was good
            if echo "${analysis_output}" | grep -q "GOOD:"; then
                print_success "Timestamp synchronization looks good"
            elif echo "${analysis_output}" | grep -q "WARNING:"; then
                print_warning "Timestamp synchronization issues detected"
            elif echo "${analysis_output}" | grep -q "Error"; then
                print_warning "Could not analyze JSON file"
            fi
        else
            print_warning "No metadata JSON files found in recent directory"
        fi
    else
        print_info "No recent recording directories found in ${DATA_VOLUME}"
    fi
}

check_hardware() {
    print_section "8. HARDWARE INFORMATION"

    print_info "PCI Devices (Ethernet/Thunderbolt adapters):"
    lspci 2>/dev/null | grep -iE 'ethernet|network|thunderbolt' | tee -a "${LOG_FILE}"

    print_info ""
    print_info "USB Devices:"
    lsusb 2>/dev/null | tee -a "${LOG_FILE}"

    # Check ethernet adapter details if ethtool is available
    if command_exists ethtool && [ -n "${NETWORK_INTERFACE}" ]; then
        print_info ""
        run_command "Ethernet Adapter Details for ${NETWORK_INTERFACE}" "sudo ethtool ${NETWORK_INTERFACE}"
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

    # Docker check
    if docker image inspect peabody124/mocap >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Docker image available" | tee -a "${LOG_FILE}"
    else
        echo -e "${RED}✗${NC} Docker image not found" | tee -a "${LOG_FILE}"
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
    print_info ""
    print_info "Send the tarball to your system administrator for analysis."
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
    if [ "$EUID" -ne 0 ]; then
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
