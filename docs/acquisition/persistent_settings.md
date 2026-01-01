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

## After Running the Script

### Laptop Mode

You no longer need to manually run:
- `set_mtu.sh` - MTU is set automatically when DHCP-Server connection activates
- `sudo service isc-dhcp-server start` - DHCP server starts automatically on boot

To start acquisition:
```bash
nmcli con up DHCP-Server
make run
```

### Network Mode

You no longer need to manually run:
- `set_mtu.sh` - MTU applies automatically on the network connection

To start acquisition:
```bash
make run
```

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

## Re-running the Script

The script is idempotent and can be safely re-run. It will:
- Skip settings that are already configured
- Update settings that have changed
- Not create duplicate entries in configuration files

## Troubleshooting

**Error: .env file not found**
- Create `.env` from `.env.template` and fill in required values

**Error: DHCP-Server connection profile not found**
- Complete the [DHCP Setup](dhcp_setup.md) before running this script

**Warning: No active connection found**
- In network mode, ensure your network interface is connected before running the script
- You may need to manually set MTU with: `nmcli con modify <connection-name> ethernet.mtu 9000`

**DHCP server not starting on boot**
- Check if isc-dhcp-server is installed: `dpkg -l | grep isc-dhcp-server`
- Check systemd service status: `systemctl status isc-dhcp-server`
- Review DHCP logs: `sudo journalctl -u isc-dhcp-server`
