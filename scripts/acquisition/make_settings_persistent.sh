#!/bin/bash

# Script to make acquisition network settings persist across reboots
# This script configures MTU, network buffers, and optionally DHCP server settings
# to survive system reboots.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Find the .env file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

# Check if .env file exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${RED}Error: .env file not found at $ENV_FILE${NC}"
    echo "Please create .env file from .env.template before running this script."
    exit 1
fi

# Source the .env file
export $(grep -v '^#' "$ENV_FILE" | grep -v '^$' | xargs)

# Check required variables
if [ -z "$NETWORK_INTERFACE" ]; then
    echo -e "${RED}Error: NETWORK_INTERFACE not set in .env file${NC}"
    exit 1
fi

# Check if DEPLOYMENT_MODE is set, default to network if not
if [ -z "$DEPLOYMENT_MODE" ]; then
    echo -e "${YELLOW}Warning: DEPLOYMENT_MODE not set in .env, defaulting to 'network'${NC}"
    DEPLOYMENT_MODE="network"
fi

echo "=========================================="
echo "Making Acquisition Settings Persistent"
echo "=========================================="
echo "Deployment mode: $DEPLOYMENT_MODE"
echo "Network interface: $NETWORK_INTERFACE"
echo ""

# 1. Make MTU persistent via NetworkManager
echo "Configuring persistent MTU settings..."

if [ "$DEPLOYMENT_MODE" = "laptop" ]; then
    # In laptop mode, modify the DHCP-Server connection profile
    if nmcli con show DHCP-Server &>/dev/null; then
        nmcli con modify DHCP-Server ethernet.mtu 9000
        echo -e "${GREEN}✓${NC} MTU set to 9000 on DHCP-Server connection"
    else
        echo -e "${RED}Error: DHCP-Server connection profile not found${NC}"
        echo "Please complete DHCP setup before running this script."
        exit 1
    fi
else
    # In network mode, try to find and modify the active connection on the interface
    ACTIVE_CON=$(nmcli -t -f NAME,DEVICE con show --active | grep "$NETWORK_INTERFACE" | cut -d: -f1)
    if [ -n "$ACTIVE_CON" ]; then
        nmcli con modify "$ACTIVE_CON" ethernet.mtu 9000
        echo -e "${GREEN}✓${NC} MTU set to 9000 on connection '$ACTIVE_CON'"
    else
        echo -e "${YELLOW}Warning: No active connection found on $NETWORK_INTERFACE${NC}"
        echo "You may need to manually set MTU with: nmcli con modify <connection-name> ethernet.mtu 9000"
    fi
fi

# 2. Make network buffer settings persistent via sysctl.conf
echo ""
echo "Configuring persistent network buffer settings..."

SYSCTL_CONF="/etc/sysctl.conf"
SETTINGS_ADDED=0

if ! grep -q "^net.core.rmem_max=10000000" "$SYSCTL_CONF" 2>/dev/null; then
    echo "net.core.rmem_max=10000000" | sudo tee -a "$SYSCTL_CONF" > /dev/null
    SETTINGS_ADDED=1
    echo -e "${GREEN}✓${NC} Added net.core.rmem_max to $SYSCTL_CONF"
else
    echo -e "${GREEN}✓${NC} net.core.rmem_max already configured"
fi

if ! grep -q "^net.core.rmem_default=10000000" "$SYSCTL_CONF" 2>/dev/null; then
    echo "net.core.rmem_default=10000000" | sudo tee -a "$SYSCTL_CONF" > /dev/null
    SETTINGS_ADDED=1
    echo -e "${GREEN}✓${NC} Added net.core.rmem_default to $SYSCTL_CONF"
else
    echo -e "${GREEN}✓${NC} net.core.rmem_default already configured"
fi

# Apply sysctl settings immediately
if [ $SETTINGS_ADDED -eq 1 ]; then
    sudo sysctl -p > /dev/null
    echo -e "${GREEN}✓${NC} Applied sysctl settings"
fi

# 3. DHCP server auto-start (laptop mode only)
if [ "$DEPLOYMENT_MODE" = "laptop" ]; then
    echo ""
    echo "Configuring DHCP server to start on boot..."

    if systemctl list-unit-files | grep -q isc-dhcp-server; then
        sudo systemctl enable isc-dhcp-server
        echo -e "${GREEN}✓${NC} DHCP server enabled for auto-start"
    else
        echo -e "${YELLOW}Warning: isc-dhcp-server not found${NC}"
        echo "Please install isc-dhcp-server if using laptop mode."
    fi
else
    echo ""
    echo "Skipping DHCP configuration (network mode)"
fi

# 4. Verify settings
echo ""
echo "=========================================="
echo "Verifying Configuration"
echo "=========================================="

# Check MTU
if [ "$DEPLOYMENT_MODE" = "laptop" ]; then
    MTU_SETTING=$(nmcli -t -f 802-3-ethernet.mtu con show DHCP-Server)
    echo "MTU (DHCP-Server): $MTU_SETTING"
else
    if [ -n "$ACTIVE_CON" ]; then
        MTU_SETTING=$(nmcli -t -f 802-3-ethernet.mtu con show "$ACTIVE_CON")
        echo "MTU ($ACTIVE_CON): $MTU_SETTING"
    fi
fi

# Check sysctl settings
echo "Network buffers:"
sysctl net.core.rmem_max net.core.rmem_default | sed 's/^/  /'

# Check DHCP server
if [ "$DEPLOYMENT_MODE" = "laptop" ]; then
    DHCP_ENABLED=$(systemctl is-enabled isc-dhcp-server 2>/dev/null || echo "not found")
    echo "DHCP server auto-start: $DHCP_ENABLED"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Configuration Complete${NC}"
echo "=========================================="
echo ""
echo "Settings will now persist across reboots."
echo ""

if [ "$DEPLOYMENT_MODE" = "laptop" ]; then
    echo "Note: You no longer need to run set_mtu.sh manually."
    echo "The DHCP server will start automatically on boot."
    echo ""
    echo "To start acquisition, activate the network profile:"
    echo "  nmcli con up DHCP-Server"
else
    echo "Note: You no longer need to run set_mtu.sh manually."
    echo ""
    echo "MTU and network buffer settings will apply automatically."
fi
