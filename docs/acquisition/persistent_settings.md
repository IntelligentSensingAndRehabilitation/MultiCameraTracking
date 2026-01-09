# Persistent Network Settings

This guide explains how to make network settings persist across reboots, eliminating the need to manually run `set_mtu.sh` and start the DHCP server each session.

## Overview

The acquisition system requires specific network settings:
- MTU of 9000 (jumbo frames) for high-bandwidth camera streaming
- Increased network buffer sizes to prevent packet drops
- DHCP server running (laptop mode only)

By default, these settings must be configured manually each session. The persistence script automates this configuration to survive reboots.

## Prerequisites

Before running the persistence script, you must complete:
1. [Docker Setup](docker_setup.md)
2. [DHCP Setup](dhcp_setup.md) (laptop mode only)
3. [Acquisition Software Setup](acquisition_software_setup.md) - ensure `.env` file is created

## Running the Persistence Script

From the root of the MultiCameraTracking repository:

```bash
./scripts/acquisition/make_settings_persistent.sh
```

The script will:
1. Read your `.env` file to determine deployment mode and network interface
2. Configure MTU to 9000 via NetworkManager (persists across reboots)
3. Add network buffer settings to `/etc/sysctl.conf` (persists across reboots)
4. Enable DHCP server auto-start on boot (laptop mode only)
5. Verify all settings are applied correctly

## What Gets Configured

### MTU Settings

**Laptop Mode:**
- Sets MTU to 9000 on the DHCP-Server NetworkManager connection profile
- MTU automatically applies when DHCP-Server connection activates

**Network Mode:**
- Sets MTU to 9000 on the currently active connection for your network interface

### Network Buffer Settings

Adds the following to `/etc/sysctl.conf`:
```
net.core.rmem_max=10000000
net.core.rmem_default=10000000
```

These settings increase the kernel's receive buffer size to handle high-bandwidth video streams from multiple cameras.

### DHCP Server (Laptop Mode Only)

Enables the `isc-dhcp-server` systemd service to start automatically on boot.

## Deployment Modes

The script behavior depends on the `DEPLOYMENT_MODE` variable in your `.env` file:

**laptop** (default):
- Cameras connect directly to laptop via network switch
- Laptop acts as DHCP server
- Requires DHCP server setup and auto-start configuration

**network**:
- Computer and cameras on existing building network
- Network infrastructure provides DHCP
- Only MTU and buffer settings are configured

## Verifying Settings

Check MTU setting:
```bash
# Laptop mode
nmcli con show DHCP-Server | grep mtu

# Network mode
ip link show <interface-name>
```

Check network buffer settings:
```bash
sysctl net.core.rmem_max net.core.rmem_default
```

Check DHCP server status (laptop mode):
```bash
systemctl is-enabled isc-dhcp-server
systemctl status isc-dhcp-server
```

## Troubleshooting

For detailed solutions, see the [Troubleshooting Guide](troubleshooting.md):

- [Environment Configuration Issues](troubleshooting.md#environment-configuration-issues)
- [MTU Configuration Issues](troubleshooting.md#mtu-configuration-issues)
- [Network Buffer Issues](troubleshooting.md#network-buffer-issues)
- [DHCP Server Issues](troubleshooting.md#dhcp-server-issues)
