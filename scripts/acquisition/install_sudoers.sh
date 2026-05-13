#!/bin/bash
################################################################################
# install_sudoers.sh
#
# Install /etc/sudoers.d/mocap-acquisition so the acquisition operator can
# run the specific commands that start_acquisition.sh's auto-remediation
# path needs (MTU/rmem fix, DHCP service start, nmcli profile activation)
# without being prompted for a password.
#
# Whitelist-only: the file allows exact binary paths and argument shapes,
# nothing else. Validates the file with `visudo -cf` before installing.
#
# Usage:
#   sudo ./scripts/acquisition/install_sudoers.sh             # for $SUDO_USER
#   sudo ./scripts/acquisition/install_sudoers.sh someuser    # for someuser
################################################################################

set -e

if [ "$EUID" -ne 0 ]; then
    echo "Error: this script must be run with sudo"
    echo "Usage: sudo $0 [username]"
    exit 1
fi

TARGET_USER="${1:-${SUDO_USER:-}}"
if [ -z "$TARGET_USER" ]; then
    echo "Error: could not determine target user (pass one as the first arg)"
    exit 1
fi
if ! id "$TARGET_USER" >/dev/null 2>&1; then
    echo "Error: user '$TARGET_USER' does not exist"
    exit 1
fi

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

print_info()    { echo "$1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error()   { echo -e "${RED}✗${NC} $1"; }

# Resolve full binary paths. sudoers entries match by absolute path, and
# distros vary (/usr/sbin vs /sbin, /usr/bin vs /bin), so detect at install
# time rather than hard-coding.
resolve_bin() {
    local name="$1"
    local path
    path=$(command -v "$name" 2>/dev/null || true)
    if [ -z "$path" ]; then
        print_error "Required binary not found in PATH: $name"
        exit 1
    fi
    echo "$path"
}

IP_BIN=$(resolve_bin ip)
SYSCTL_BIN=$(resolve_bin sysctl)
SYSTEMCTL_BIN=$(resolve_bin systemctl)
NMCLI_BIN=$(resolve_bin nmcli)

SUDOERS_PATH="/etc/sudoers.d/mocap-acquisition"
TMP_FILE=$(mktemp)
trap 'rm -f "$TMP_FILE"' EXIT

cat > "$TMP_FILE" <<EOF
# /etc/sudoers.d/mocap-acquisition
# Installed by scripts/acquisition/install_sudoers.sh.
# Whitelists the commands that start_acquisition.sh's auto-remediation
# path runs via 'sudo -n'. Removing this file disables auto-remediation
# but does not break the acquisition stack itself.

$TARGET_USER ALL=(root) NOPASSWD: $IP_BIN link set * mtu *
$TARGET_USER ALL=(root) NOPASSWD: $SYSCTL_BIN -w net.core.rmem_max=*
$TARGET_USER ALL=(root) NOPASSWD: $SYSTEMCTL_BIN start isc-dhcp-server
$TARGET_USER ALL=(root) NOPASSWD: $SYSTEMCTL_BIN restart isc-dhcp-server
$TARGET_USER ALL=(root) NOPASSWD: $NMCLI_BIN con up DHCP-Server
EOF

echo ""
echo -e "${CYAN}=== sudoers content to install ===${NC}"
cat "$TMP_FILE"
echo ""

# Validate before touching /etc/sudoers.d. A syntactically broken file
# there will lock everyone out of sudo on this host.
if ! visudo -cf "$TMP_FILE" >/dev/null; then
    print_error "visudo rejected the generated file — refusing to install"
    visudo -cf "$TMP_FILE" || true
    exit 1
fi
print_success "visudo validated the file"

install -m 0440 -o root -g root "$TMP_FILE" "$SUDOERS_PATH"
print_success "Installed $SUDOERS_PATH for user '$TARGET_USER'"
echo ""

# Confirm the user can actually exercise one of the rules without a
# password. `sudo -l` lists permitted commands for the current effective
# user; under `sudo -u $TARGET_USER` that's the target.
if sudo -u "$TARGET_USER" -n -l "$SYSTEMCTL_BIN" start isc-dhcp-server >/dev/null 2>&1; then
    print_success "Verified passwordless sudo for systemctl start isc-dhcp-server"
else
    print_warning "Could not verify passwordless sudo for the new rules"
    print_info "  Run this manually as '$TARGET_USER':"
    print_info "    sudo -n -l $SYSTEMCTL_BIN start isc-dhcp-server"
fi
