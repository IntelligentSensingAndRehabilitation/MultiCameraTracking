#!/bin/bash

################################################################################
# MultiCameraTracking Host Health Remediation
#
# Purpose: Apply safe host-side fixes (MTU, network buffers, DHCP server) that
#          the in-container `make health` check cannot perform itself — the
#          acquisition container has no sudo and cannot reach host systemd.
#          Runs the same `sudo -n` remediations start_acquisition.sh applies at
#          startup, but standalone, so an operator can fix host drift mid-session
#          without restarting the acquisition stack. Finishes by running the
#          container health check so the operator sees the full post-fix state
#          (including camera reachability, which this script does not touch).
#
# Usage: invoked via `make health-fix`. Requires passwordless sudo for the
#        remediation commands — run `make install-sudoers` once to set that up.
################################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

print_header() {
    echo ""
    echo -e "${BOLD}${CYAN}========================================${NC}"
    echo -e "${BOLD}${CYAN}$1${NC}"
    echo -e "${BOLD}${CYAN}========================================${NC}"
    echo ""
}

print_step() { echo -e "${CYAN}[$1]${NC} $2"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_info() { echo "  $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
cd "$REPO_ROOT"

if [ ! -f "$ENV_FILE" ]; then
    print_error ".env file not found at $ENV_FILE"
    print_info "Create .env from .env.template"
    exit 1
fi
export $(grep -v '^#' "$ENV_FILE" | grep -v '^$' | xargs)

if [ -z "$NETWORK_INTERFACE" ]; then
    print_error "NETWORK_INTERFACE not set in .env"
    exit 1
fi
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-laptop}"

print_header "Host Remediation"
print_info "Interface: $NETWORK_INTERFACE   Mode: $DEPLOYMENT_MODE"

# MTU: jumbo frames are required for GigE camera throughput.
print_step "1/4" "MTU"
if ip link show "$NETWORK_INTERFACE" &>/dev/null; then
    mtu=$(ip link show "$NETWORK_INTERFACE" | grep -oP 'mtu \K\d+')
    if [ "$mtu" = "9000" ]; then
        print_success "MTU already 9000"
    elif sudo -n ip link set "$NETWORK_INTERFACE" mtu 9000 2>/dev/null; then
        print_success "Set MTU to 9000 (was $mtu)"
    else
        print_error "Could not set MTU (passwordless sudo unavailable?)"
        print_info "Run: sudo ip link set $NETWORK_INTERFACE mtu 9000"
        print_info "Or persist via: ./scripts/acquisition/make_settings_persistent.sh"
    fi
else
    print_warning "Interface $NETWORK_INTERFACE not present — skipping MTU"
fi

# rmem_max: large socket receive buffers prevent dropped frames under load.
print_step "2/4" "Network buffers (rmem_max)"
rmem_max=$(sysctl -n net.core.rmem_max 2>/dev/null)
if [ -n "$rmem_max" ] && [ "$rmem_max" -ge 10000000 ]; then
    print_success "rmem_max already $rmem_max"
elif sudo -n sysctl -w net.core.rmem_max=10000000 >/dev/null 2>&1; then
    print_success "Set net.core.rmem_max to 10000000 (was ${rmem_max:-unset})"
else
    print_error "Could not set rmem_max (passwordless sudo unavailable?)"
    print_info "Run: sudo sysctl -w net.core.rmem_max=10000000"
    print_info "Or persist via: ./scripts/acquisition/make_settings_persistent.sh"
fi

# DHCP: laptop mode is the camera network's only DHCP server. In network mode
# DHCP is upstream and there is nothing local to start.
print_step "3/4" "DHCP server"
if [ "$DEPLOYMENT_MODE" != "laptop" ]; then
    print_info "Network mode — DHCP is upstream, nothing to remediate"
elif systemctl is-active --quiet isc-dhcp-server 2>/dev/null; then
    print_success "isc-dhcp-server already running"
elif sudo -n systemctl start isc-dhcp-server >/dev/null 2>&1 \
    && systemctl is-active --quiet isc-dhcp-server 2>/dev/null; then
    print_success "Started isc-dhcp-server"
else
    print_error "Could not start isc-dhcp-server (passwordless sudo unavailable?)"
    print_info "Start manually: sudo systemctl start isc-dhcp-server"
    print_info "Check logs: journalctl -u isc-dhcp-server -n 50"
    print_info "Or enable on boot: ./scripts/acquisition/make_settings_persistent.sh"
fi

# lldpd: switch discovery (LLDP) requires this service.
print_step "4/4" "lldpd (switch discovery)"
if ! command -v lldpcli &>/dev/null; then
    print_warning "lldpd not installed — switch discovery unavailable"
    print_info "Install with: sudo apt install lldpd"
elif systemctl is-active --quiet lldpd 2>/dev/null; then
    print_success "lldpd already running"
elif sudo -n systemctl start lldpd >/dev/null 2>&1 \
    && systemctl is-active --quiet lldpd 2>/dev/null; then
    print_success "Started lldpd"
else
    print_error "Could not start lldpd (passwordless sudo unavailable?)"
    print_info "Start manually: sudo systemctl start lldpd"
fi

print_header "Health Report (post-remediation)"
exec make health
