# Automated Setup Wizard

The setup wizard automates the complete acquisition system setup process from a fresh Ubuntu installation to a fully configured system ready for use.

## Overview

The wizard:
- Validates system prerequisites
- Installs Docker and dependencies
- Configures DHCP server (laptop mode)
- Sets up network interfaces
- Creates required directories
- Generates environment configuration
- Applies persistent settings
- Builds Docker images

**Time required:** 15-30 minutes (mostly Docker image building)

## Prerequisites

Before running the setup wizard:

1. **Fresh Ubuntu 22.04 LTS installation**
2. **Repository cloned:**
   ```bash
   git clone https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking
   cd MultiCameraTracking
   ```
3. **Network adapter connected** (for laptop mode)
4. **Switch MAC address** (for laptop mode) - found on switch hardware label

## Running the Setup Wizard

From the MultiCameraTracking repository root:

```bash
sudo ./scripts/acquisition/setup_acquisition_system.sh
```

The wizard must run with sudo to install packages and configure system settings.

## Setup Process

The wizard guides you through 9 steps:

### Step 1: System Prerequisites Check

Validates your system meets requirements:
- Ubuntu 22.04 (recommended)
- Kernel version check
- RAM (minimum 32GB recommended)
- CPU cores (minimum 20+ recommended)
- Repository structure

If any check fails, you can choose to continue anyway or cancel setup.

### Step 2: Deployment Mode Selection

Choose how the system will be deployed:

**Option 1: Laptop mode (portable system with DHCP server)**
- Laptop acts as DHCP server for cameras
- Cameras connect via network switch to laptop
- Most common setup
- Requires DHCP server configuration

**Option 2: Network mode (building network infrastructure)**
- Computer and cameras on existing network
- Network already provides DHCP/routing
- Less common
- Skips DHCP configuration steps

**Selection:** Choose [1] or [2]

### Step 3: Docker Installation

Checks if Docker is installed. If not:
- Prompts for installation confirmation
- Adds Docker repository
- Installs Docker Engine and Docker Compose
- Adds user to docker group
- Starts Docker service

If Docker is already installed, checks docker group membership.

**Note:** You'll need to log out and back in for docker group changes to take effect.

### Step 4: Network Interface Detection

Auto-detects available ethernet interfaces and displays:
- Interface name (e.g., enp5s0)
- MAC address
- Connection status

**Selection:**
- If only one interface found: Auto-selected
- If multiple interfaces: Choose from numbered list

### Step 5: DHCP Server Setup (Laptop Mode Only)

Skipped in network mode.

In laptop mode:
- Installs isc-dhcp-server package
- Auto-detects MAC address of selected network interface
- Generates `/etc/dhcp/dhcpd.conf` with proper subnet configuration
- Configures `/etc/default/isc-dhcp-server`
- Creates NetworkManager "DHCP-Server" profile

**DHCP Configuration:**
- Subnet: 192.168.1.0/24
- Laptop IP: 192.168.1.1 (fixed reservation using detected MAC)
- Camera range: 192.168.1.10 - 192.168.1.100
- DNS: 8.8.8.8, 8.8.4.4

### Step 6: Directory Creation

Prompts for directory locations with sensible defaults:

**Data storage directory** (default: /data)
- Where recordings will be saved
- Requires sufficient free space

**Camera configs directory** (default: /camera_configs)
- Where camera YAML configuration files are stored

**DataJoint external storage** (default: /mnt/datajoint_external)
- Optional directory for DataJoint database external storage
- Can skip if not using DataJoint

The wizard creates directories if they don't exist and sets proper ownership.

### Step 7: Environment Configuration

Creates `.env` file with your configuration.

**Database settings:**
- Username (default: root)
- Password (default: pose)
- Host (default: 127.0.0.1)
- Port (default: 3306)

**Other settings:**
- React app base URL (default: localhost)
- Disk space warning threshold in GB (default: 50)

All previously configured values (deployment mode, network interface, directories) are automatically included.

If `.env` already exists, prompts before overwriting.

### Step 8: Persistent Network Settings

Automatically runs the persistence script to make settings survive reboots:
- MTU set to 9000 via NetworkManager
- Network buffers configured in /etc/sysctl.conf
- DHCP server enabled on boot (laptop mode only)

**Prompt:** Apply persistent settings now? [Y/n]

If skipped, you can run manually later:
```bash
./scripts/acquisition/make_settings_persistent.sh
```

### Step 9: Docker Image Build

Builds the mocap Docker image required for acquisition.

**Warning:** This step takes 10-20 minutes depending on your system.

**Prompt:** Build Docker image now? [Y/n]

If skipped, you must build before first use:
```bash
make build-mocap
```

## After Setup

### Immediate Next Steps

1. **Activate docker group** (choose one):
   - **Option A (immediate):** Run `newgrp docker` to activate in current shell
   - **Option B (permanent):** Log out and log back in
   - Verify with: `groups` (should show "docker")

2. **Create camera configuration files**
   - Location: The camera configs directory you specified (e.g., /camera_configs)
   - Format: YAML files with camera serial numbers and settings
   - See: [Example Config](example_config.md)

3. **Hardware setup:**
   - **Laptop mode:** Connect cameras to network switch, switch to laptop
   - **Network mode:** Ensure cameras are on the building network

4. **Start acquisition:**
   ```bash
   ./scripts/acquisition/start_acquisition.sh
   ```
   Or:
   ```bash
   make start-acquisition
   ```

### Verification

Verify your setup with the diagnostics script:

```bash
sudo ./scripts/acquisition/acquisition_diagnostics.sh
```

Should show:
- 0 errors
- Network configured correctly
- DHCP server running (laptop mode)
- Directories created

## Troubleshooting

### Error: Must run with sudo

The setup script needs root privileges to install packages and configure system settings.

```bash
sudo ./scripts/acquisition/setup_acquisition_system.sh
```

### Error: Repository appears incomplete

Ensure you're in the MultiCameraTracking repository root and all files were cloned correctly.

```bash
cd MultiCameraTracking
git pull  # Update to latest
```

### Warning: System doesn't meet prerequisites

Common warnings:
- **RAM < 32GB:** Can still work but may have performance issues with 8-10 cameras
- **CPU cores < 20:** Can still work but frame processing may be slower
- **Not Ubuntu 22.04:** May work on other versions but is untested

You can continue setup despite warnings, but functionality may be limited.

### Docker group membership not working

After setup completes, activate the docker group:

**Option 1 (recommended for testing):**
```bash
newgrp docker
```
This activates the group immediately in your current shell.

**Option 2 (permanent):**
1. Log out completely
2. Log back in
3. Verify with: `groups` (should show "docker")

### Could not detect MAC address

If the wizard fails to detect the MAC address of your network interface:

- Ensure the network cable is plugged in
- Try `ip link show <interface-name>` manually to verify it has a MAC address
- The interface must be a physical ethernet adapter (not virtual)

### Docker build failed

Common causes:
- Insufficient disk space (need ~10GB free)
- Network issues downloading dependencies
- Missing download_flir.sh execution

Fix and retry:
```bash
cd docker
./download_flir.sh  # If you haven't run this
cd ..
make build-mocap
```

### DHCP server won't start

After setup, check:
```bash
sudo systemctl status isc-dhcp-server
sudo journalctl -u isc-dhcp-server
```

Common issues:
- Network interface not connected
- Invalid DHCP configuration
- NetworkManager profile not activated

## Manual Setup Alternative

If you prefer manual setup or the wizard doesn't work for your environment, follow the individual setup guides:

1. [General System Setup](general_system_setup.md)
2. [Docker Setup](docker_setup.md)
3. [DHCP Server Setup](dhcp_setup.md) (laptop mode only)
4. [Acquisition Software Setup](acquisition_software_setup.md)
5. [Persistent Settings](persistent_settings.md)

## What the Wizard Doesn't Do

The wizard automates system setup but **does not:**

1. **Create camera configuration files**
   - You must create these manually for your specific cameras
   - See [Example Config](example_config.md)

2. **Install Spinnaker GUI application**
   - Only needed if you want to use Spinnaker's native GUI
   - See [Spinnaker App Setup](spinnaker_app_setup.md)

3. **Configure kernel pinning**
   - Recommended to prevent auto-updates breaking camera drivers
   - See [General System Setup](general_system_setup.md)

4. **Download FLIR SDK**
   - Should be run separately before Docker build
   - Script: `docker/download_flir.sh`

## Unattended Setup

For automated deployment, you can pre-configure answers (though not recommended for first-time setup):

```bash
# Example: Fully automated setup (dangerous - no validation)
# This is just to show the concept - manual interaction is safer

echo "1" | sudo ./scripts/acquisition/setup_acquisition_system.sh
```

However, we recommend interactive mode for proper validation and error checking.

## Getting Help

If you encounter issues during setup:

1. **Review error messages** - the wizard provides specific guidance
2. **Run diagnostics:**
   ```bash
   sudo ./scripts/acquisition/acquisition_diagnostics.sh
   ```
3. **Check individual setup guides** for detailed manual steps
4. **Report issues** with diagnostic output at:
   https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/issues
