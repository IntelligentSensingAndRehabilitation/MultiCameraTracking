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

## Running the Setup Wizard

From the MultiCameraTracking repository root:

```bash
sudo ./scripts/acquisition/setup_acquisition_system.sh
```

The wizard must run with sudo to install packages and configure system settings.

## Detailed Steps

The wizard guides you through 9 steps:

### Step 1: System Prerequisites Check

Validates your system meets requirements

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

### Step 3: Docker Installation

Checks if Docker is installed. If not:
- Prompts for installation confirmation
- Adds Docker repository
- Installs Docker Engine and Docker Compose
- Adds user to docker group
- Starts Docker service

**Note:** You'll need to open a new terminal for docker group changes to take effect.

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

### Step 9: Docker Image Build

Builds the mocap Docker image required for acquisition.

## After Setup

1. **Activate docker group** (choose one):
   - **Option A (immediate):** Run `newgrp docker` to activate in current shell
   - **Option B (permanent):** Log out and log back in
   - Verify with: `groups` (should show "docker")

2. **Create camera configuration files**
   - Location: The camera configs directory you specified (e.g., /camera_configs)
   - Format: YAML files with camera serial numbers and settings
   - See: [Example Config](example_config.md)

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

## Getting Help

If you encounter issues during setup:

1. **Review error messages** - the wizard provides specific guidance
2. **Check individual setup guides** for detailed manual steps
3. **Report issues** with diagnostic output at:
   https://github.com/IntelligentSensingAndRehabilitation/MultiCameraTracking/issues
