# Persistent Network Settings

This guide explains how to make network settings persist across reboots, eliminating the need to manually run `set_mtu.sh` and start the DHCP server each session.

## Overview

The acquisition system requires specific network settings:
- MTU of 9000 (jumbo frames) for high-bandwidth camera streaming
- Increased network buffer sizes to prevent packet drops
- DHCP server running (laptop mode only)

By default, these settings must be configured manually each session. The persistence script automates this configuration to survive reboots.

This persistence script is included as part of the automated setup wizard (recommended).

## Prerequisites

If the persistence script needs to be run manually, ensure the acquisition system is fully set up beforehand.

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

## Troubleshooting

For detailed solutions, see the [Troubleshooting Guide](troubleshooting.md):

- [MTU Configuration Issues](troubleshooting.md#mtu-configuration-issues)
- [Network Buffer Issues](troubleshooting.md#network-buffer-issues)
- [DHCP Server Issues](troubleshooting.md#dhcp-server-issues)
