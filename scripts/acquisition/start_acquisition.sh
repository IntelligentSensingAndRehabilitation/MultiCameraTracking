#!/bin/bash

################################################################################
# MultiCameraTracking Unified Startup Script
#
# Purpose: Simplified startup script with system validation
# Usage: ./start_acquisition.sh [--skip-checks]
#        --skip-checks: Skip system validation (not recommended)
#
# This script automates the acquisition system startup process with validation
# to catch configuration issues before they cause problems.
################################################################################

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Parse command line arguments
SKIP_CHECKS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-checks)
            SKIP_CHECKS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--skip-checks]"
            echo ""
            echo "Options:"
            echo "  --skip-checks    Skip system validation checks"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Find the repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

# Change to repository root
cd "$REPO_ROOT"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo ""
    echo -e "${BOLD}${CYAN}========================================${NC}"
    echo -e "${BOLD}${CYAN}$1${NC}"
    echo -e "${BOLD}${CYAN}========================================${NC}"
    echo ""
}

print_step() {
    echo -e "${CYAN}[$1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo "  $1"
}

################################################################################
# System Checks
################################################################################

run_system_checks() {
    print_header "System Checks"

    local checks_passed=true

    # Check 1: .env file exists
    print_step "1/7" "Checking .env file"
    if [ ! -f "$ENV_FILE" ]; then
        print_error ".env file not found at $ENV_FILE"
        print_info "Create .env from .env.template"
        return 1
    fi
    print_success ".env file found"

    # Source .env file
    export $(grep -v '^#' "$ENV_FILE" | grep -v '^$' | xargs)

    # Check 2: Required environment variables
    print_step "2/7" "Checking required environment variables"
    local missing_vars=()

    [ -z "$NETWORK_INTERFACE" ] && missing_vars+=("NETWORK_INTERFACE")
    [ -z "$DATA_VOLUME" ] && missing_vars+=("DATA_VOLUME")
    [ -z "$CAMERA_CONFIGS" ] && missing_vars+=("CAMERA_CONFIGS")

    if [ ${#missing_vars[@]} -gt 0 ]; then
        print_error "Missing required variables in .env: ${missing_vars[*]}"
        checks_passed=false
    else
        print_success "All required environment variables set"
    fi

    # Set default deployment mode if not specified
    if [ -z "$DEPLOYMENT_MODE" ]; then
        DEPLOYMENT_MODE="laptop"
        print_warning "DEPLOYMENT_MODE not set, defaulting to 'laptop'"
    else
        print_info "Deployment mode: $DEPLOYMENT_MODE"
    fi

    # Check 3: Network interface exists
    print_step "3/7" "Checking network interface"
    if ! ip link show "$NETWORK_INTERFACE" &>/dev/null; then
        print_error "Network interface $NETWORK_INTERFACE not found"
        print_info "Available interfaces:"
        ip link show | grep -E '^[0-9]+:' | awk '{print "  - " $2}' | sed 's/:$//'
        checks_passed=false
    else
        print_success "Network interface $NETWORK_INTERFACE found"
    fi

    # Check 4: MTU setting (auto-remediate before failing).
    print_step "4/7" "Checking MTU setting"
    if ip link show "$NETWORK_INTERFACE" &>/dev/null; then
        local mtu=$(ip link show "$NETWORK_INTERFACE" | grep -oP 'mtu \K\d+')
        if [ "$mtu" = "9000" ]; then
            print_success "MTU set to 9000"
        else
            print_warning "MTU is $mtu, attempting auto-fix"
            if sudo -n ip link set "$NETWORK_INTERFACE" mtu 9000 2>/dev/null; then
                local mtu_after=$(ip link show "$NETWORK_INTERFACE" | grep -oP 'mtu \K\d+')
                if [ "$mtu_after" = "9000" ]; then
                    print_success "Auto-fixed MTU to 9000 (was $mtu)"
                else
                    print_error "MTU still $mtu_after after auto-fix attempt"
                    print_info "Run: ./scripts/acquisition/make_settings_persistent.sh"
                    checks_passed=false
                fi
            else
                print_error "MTU is $mtu, expected 9000 — auto-fix failed (passwordless sudo unavailable?)"
                print_info "Run: sudo ip link set $NETWORK_INTERFACE mtu 9000"
                print_info "Or persist via: ./scripts/acquisition/make_settings_persistent.sh"
                checks_passed=false
            fi
        fi
    fi

    # Check 5: Network buffers (auto-remediate before failing).
    print_step "5/7" "Checking network buffer settings"
    local rmem_max=$(sysctl -n net.core.rmem_max 2>/dev/null)
    if [ -n "$rmem_max" ] && [ "$rmem_max" -ge 10000000 ]; then
        print_success "Network buffers configured correctly"
    else
        print_warning "rmem_max is ${rmem_max:-unset}, attempting auto-fix"
        if sudo -n sysctl -w net.core.rmem_max=10000000 >/dev/null 2>&1; then
            local rmem_after=$(sysctl -n net.core.rmem_max 2>/dev/null)
            if [ -n "$rmem_after" ] && [ "$rmem_after" -ge 10000000 ]; then
                print_success "Auto-fixed net.core.rmem_max to $rmem_after"
            else
                print_error "rmem_max still $rmem_after after auto-fix attempt"
                print_info "Run: ./scripts/acquisition/make_settings_persistent.sh"
                checks_passed=false
            fi
        else
            print_error "rmem_max is ${rmem_max:-unset} — auto-fix failed (passwordless sudo unavailable?)"
            print_info "Run: sudo sysctl -w net.core.rmem_max=10000000"
            print_info "Or persist via: ./scripts/acquisition/make_settings_persistent.sh"
            checks_passed=false
        fi
    fi

    # Check 6: Disk space
    print_step "6/7" "Checking disk space"
    if [ -d "$DATA_VOLUME" ]; then
        local available_gb=$(df -BG "$DATA_VOLUME" | tail -1 | awk '{print $4}' | sed 's/G//')
        local threshold=${DISK_SPACE_WARNING_THRESHOLD_GB:-10}

        if [ "$available_gb" -gt "$threshold" ]; then
            print_success "Disk space available: ${available_gb}GB"
        else
            print_error "Low disk space: ${available_gb}GB available (threshold: ${threshold}GB)"
            print_info "Free up space in $DATA_VOLUME before recording"
            checks_passed=false
        fi
    else
        print_warning "Data volume directory not found: $DATA_VOLUME"
        print_info "Will be created by Docker if needed"
    fi

    # Check 7: Camera configs directory
    print_step "7/7" "Checking camera configs directory"
    if [ -d "$CAMERA_CONFIGS" ]; then
        local config_count=$(find "$CAMERA_CONFIGS" -name "*.yaml" -o -name "*.yml" 2>/dev/null | wc -l)
        if [ "$config_count" -gt 0 ]; then
            print_success "Camera configs directory exists ($config_count config files found)"
        else
            print_warning "No camera config files found in $CAMERA_CONFIGS"
            print_info "You'll need to create a config file before recording"
        fi
    else
        print_warning "Camera configs directory not found: $CAMERA_CONFIGS"
        print_info "Will be created by Docker if needed"
    fi

    echo ""

    if ! $checks_passed; then
        return 1
    fi

    return 0
}

################################################################################
# Network Activation (Laptop Mode)
################################################################################

activate_network() {
    if [ "$DEPLOYMENT_MODE" != "laptop" ]; then
        print_info "Network mode detected - skipping DHCP-Server activation"
        return 0
    fi

    print_header "Network Activation"

    # Check if DHCP-Server profile exists
    if ! nmcli con show "DHCP-Server" &>/dev/null; then
        print_error "DHCP-Server connection profile not found"
        print_info "Please complete DHCP setup first"
        print_info "See: docs/acquisition/dhcp_setup.md"
        return 1
    fi

    # Check if already active
    if nmcli con show --active | grep -q "DHCP-Server"; then
        print_success "DHCP-Server profile already active"
    else
        print_step "1/2" "Activating DHCP-Server network profile"
        if nmcli con up "DHCP-Server" &>/dev/null; then
            print_success "DHCP-Server profile activated"
        else
            print_error "Failed to activate DHCP-Server profile"
            print_info "Try manually: nmcli con up DHCP-Server"
            return 1
        fi
    fi

    # Wait a moment for network to stabilize
    sleep 1

    # Verify IP address
    local ip_addr=$(ip addr show "$NETWORK_INTERFACE" | grep -oP 'inet \K[\d.]+' | head -1)
    if [ "$ip_addr" = "192.168.1.1" ]; then
        print_success "Network interface has correct IP: $ip_addr"
    else
        print_warning "Network interface IP is $ip_addr (expected 192.168.1.1)"
    fi

    # Check DHCP server status. In laptop mode there is no fallback DHCP
    # server upstream — if isc-dhcp-server isn't running, no cameras will
    # boot. Auto-remediate before failing.
    print_step "2/2" "Checking DHCP server status"
    if systemctl is-active --quiet isc-dhcp-server 2>/dev/null; then
        print_success "DHCP server is running"
    else
        print_warning "DHCP server not running, attempting to start it"
        if sudo -n systemctl start isc-dhcp-server >/dev/null 2>&1; then
            sleep 1
            if systemctl is-active --quiet isc-dhcp-server 2>/dev/null; then
                print_success "Auto-started isc-dhcp-server"
            else
                print_error "isc-dhcp-server still not running after start attempt"
                print_info "Check logs: journalctl -u isc-dhcp-server -n 50"
                return 1
            fi
        else
            print_error "Could not start isc-dhcp-server (passwordless sudo unavailable?)"
            print_info "Start manually: sudo systemctl start isc-dhcp-server"
            print_info "Or enable on boot: ./scripts/acquisition/make_settings_persistent.sh"
            return 1
        fi
    fi

    echo ""
}

################################################################################
# Camera Detection
################################################################################

wait_for_cameras() {
    print_header "Camera Detection"

    print_info "Waiting for cameras to connect (timeout: 30 seconds)"
    print_info "Cameras should show double-blink green LED when ready"
    echo ""

    # We'll check for cameras using a simple approach
    # The actual camera detection is done by the acquisition software
    local timeout=30
    local elapsed=0

    while [ $elapsed -lt $timeout ]; do
        # Simple check: are there any devices on the 192.168.1.x network responding?
        # This is a basic check - the actual acquisition software does more thorough detection

        if [ "$DEPLOYMENT_MODE" = "laptop" ]; then
            # Count current leases only — the previous `grep -c '^lease'` counted
            # stale leases from previous boots and reported false-positive counts.
            if [ -f /var/lib/dhcp/dhcpd.leases ]; then
                local lease_count=$(python3 - /var/lib/dhcp/dhcpd.leases <<'PY' 2>/dev/null || echo 0
import datetime, re, sys
try:
    text = open(sys.argv[1]).read()
except OSError:
    print(0); sys.exit(0)
now = datetime.datetime.utcnow()
n = 0
for m in re.finditer(r"lease\s+\d+\.\d+\.\d+\.\d+\s*\{(.*?)\}", text, re.DOTALL):
    body = m.group(1)
    if "abandoned" in body:
        continue
    em = re.search(r"ends\s+\d+\s+(\d{4})/(\d+)/(\d+)\s+(\d+):(\d+):(\d+)", body)
    if em is None:
        if re.search(r"ends\s+never", body):
            n += 1
        continue
    try:
        if datetime.datetime(*map(int, em.groups())) > now:
            n += 1
    except ValueError:
        pass
print(n)
PY
)
                if [ "$lease_count" -gt 0 ]; then
                    print_success "DHCP has issued $lease_count current lease(s)"
                    break
                fi
            fi
        fi

        # Simple progress indicator
        echo -ne "\rWaiting... ${elapsed}s / ${timeout}s"
        sleep 2
        elapsed=$((elapsed + 2))
    done

    echo ""
    echo ""

    if [ $elapsed -ge $timeout ]; then
        print_warning "Camera detection timeout reached"
        print_info "Cameras may still connect after startup"
    else
        print_success "Network activity detected"
    fi

    print_info "The acquisition GUI will show connected cameras once started"
    echo ""
}

################################################################################
# Docker Startup
################################################################################

start_acquisition() {
    print_header "Starting Acquisition Software"

    # Check if Docker is installed
    if ! command -v docker &>/dev/null; then
        print_error "Docker is not installed"
        print_info "See: docs/acquisition/docker_setup.md"
        return 1
    fi

    # Check if Docker daemon is running
    if ! docker info &>/dev/null; then
        print_error "Docker daemon is not running"
        print_info "Start with: sudo systemctl start docker"
        return 1
    fi

    print_success "Docker is ready"
    echo ""

    # Check if mocap image exists
    if ! docker images | grep -q mocap; then
        print_warning "mocap Docker image not found"
        print_info "Building Docker image (this may take several minutes)..."
        echo ""
        make build-mocap
        echo ""
    fi

    print_info "Starting acquisition system..."
    print_info "The browser will open at http://localhost:3000"
    echo ""
    print_info "Press Ctrl+C to stop the acquisition system"
    echo ""

    # Start the acquisition system using internal Docker target
    make _docker-run
}

################################################################################
# Main Script
################################################################################

main() {
    echo ""
    echo -e "${BOLD}${CYAN}╔════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║  MultiCameraTracking Acquisition Start ║${NC}"
    echo -e "${BOLD}${CYAN}╔════════════════════════════════════════╝${NC}"

    # Validate .env against .env.template; prompt for any missing/empty vars.
    # In --skip-checks mode, only verify (don't prompt) — the operator opted out.
    if $SKIP_CHECKS; then
        "$SCRIPT_DIR/check_env.sh" --quiet || exit 1
    else
        "$SCRIPT_DIR/check_env.sh"
    fi

    # Run system checks
    if ! $SKIP_CHECKS; then
        if ! run_system_checks; then
            echo ""
            print_error "System checks failed"
            print_info "Fix the issues above and try again"
            print_info "Or use --skip-checks to bypass (not recommended)"
            echo ""
            exit 1
        fi
    else
        print_warning "Skipping system checks (--skip-checks enabled)"
        echo ""
        # Still need to source .env
        if [ -f "$ENV_FILE" ]; then
            export $(grep -v '^#' "$ENV_FILE" | grep -v '^$' | xargs)
            DEPLOYMENT_MODE=${DEPLOYMENT_MODE:-laptop}
        fi
    fi

    # Activate network (laptop mode only)
    if ! activate_network; then
        echo ""
        print_error "Network activation failed"
        echo ""
        exit 1
    fi

    # Wait for cameras (optional, non-blocking)
    if [ "$DEPLOYMENT_MODE" = "laptop" ]; then
        wait_for_cameras
    fi

    # Start acquisition software
    echo ""
    print_success "All checks passed - ready to start acquisition"
    echo ""

    start_acquisition
}

# Run main function
main
