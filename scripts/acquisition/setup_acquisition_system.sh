#!/bin/bash

################################################################################
# MultiCameraTracking Acquisition System Setup Wizard
#
# Purpose: Automated setup wizard for initial acquisition system configuration
# Usage: sudo ./setup_acquisition_system.sh
#
# This script guides you through the complete setup process, from installing
# dependencies to configuring network settings and creating the environment.
################################################################################

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "Error: This script must be run with sudo"
    echo "Usage: sudo ./setup_acquisition_system.sh"
    exit 1
fi

# Get the actual user (not root when using sudo)
ACTUAL_USER="${SUDO_USER:-$USER}"
ACTUAL_HOME=$(getent passwd "$ACTUAL_USER" | cut -d: -f6)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Find repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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
    echo "$1"
}

ask_yes_no() {
    local question="$1"
    local default="${2:-n}"

    if [ "$default" = "y" ]; then
        prompt="[Y/n]"
    else
        prompt="[y/N]"
    fi

    while true; do
        read -p "$question $prompt: " answer
        answer=${answer:-$default}
        case "$answer" in
            [Yy]*) return 0 ;;
            [Nn]*) return 1 ;;
            *) echo "Please answer yes or no." ;;
        esac
    done
}

ask_input() {
    local question="$1"
    local default="$2"
    local answer

    if [ -n "$default" ]; then
        read -p "$question [$default]: " answer
        answer=${answer:-$default}
    else
        read -p "$question: " answer
    fi

    echo "$answer"
}

################################################################################
# Step 1: System Prerequisites Check
################################################################################

check_prerequisites() {
    print_header "Step 1: System Prerequisites Check"

    local all_checks_passed=true

    # Check Ubuntu version
    print_info "Checking Ubuntu version..."
    if [ -f /etc/os-release ]; then
        source /etc/os-release
        if [ "$ID" = "ubuntu" ] && [ "$VERSION_ID" = "22.04" ]; then
            print_success "Ubuntu 22.04 detected"
        else
            print_warning "Running $PRETTY_NAME (recommended: Ubuntu 22.04)"
        fi
    else
        print_warning "Could not detect OS version"
    fi

    # Check kernel version
    print_info "Checking kernel version..."
    KERNEL_VERSION=$(uname -r)
    print_info "  Kernel: $KERNEL_VERSION"

    # Check RAM
    print_info "Checking system RAM..."
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$TOTAL_RAM" -ge 32 ]; then
        print_success "RAM: ${TOTAL_RAM}GB (meets 32GB minimum)"
    else
        print_warning "RAM: ${TOTAL_RAM}GB (recommended: 32GB+)"
        all_checks_passed=false
    fi

    # Check CPU cores
    print_info "Checking CPU cores..."
    CPU_CORES=$(nproc)
    if [ "$CPU_CORES" -ge 20 ]; then
        print_success "CPU cores: $CPU_CORES (meets 20+ recommendation)"
    else
        print_warning "CPU cores: $CPU_CORES (recommended: 20+)"
    fi

    # Check if repository is complete
    print_info "Checking repository structure..."
    if [ -f "docker-compose.yml" ] && [ -f ".env.template" ]; then
        print_success "Repository structure looks good"
    else
        print_error "Repository appears incomplete (missing docker-compose.yml or .env.template)"
        all_checks_passed=false
    fi

    echo ""
    if ! $all_checks_passed; then
        print_warning "Some prerequisites are not met"
        if ! ask_yes_no "Continue anyway?" "n"; then
            print_info "Setup cancelled"
            exit 1
        fi
    fi

    print_success "Prerequisites check complete"
    echo ""
}

################################################################################
# Step 2: Choose Deployment Mode
################################################################################

choose_deployment_mode() {
    print_header "Step 2: Deployment Mode Selection"

    print_info "How will the acquisition system be deployed?"
    echo ""
    print_info "  [1] Laptop mode (portable system with DHCP server)"
    print_info "      - Laptop acts as DHCP server for cameras"
    print_info "      - Cameras connect via network switch to laptop"
    print_info "      - Most common setup"
    echo ""
    print_info "  [2] Network mode (building network infrastructure)"
    print_info "      - Computer and cameras on existing network"
    print_info "      - Network already provides DHCP/routing"
    print_info "      - Less common"
    echo ""

    while true; do
        read -p "Select deployment mode [1/2]: " mode_choice
        case "$mode_choice" in
            1)
                DEPLOYMENT_MODE="laptop"
                print_success "Selected: Laptop mode (with DHCP server)"
                break
                ;;
            2)
                DEPLOYMENT_MODE="network"
                print_success "Selected: Network mode (building network)"
                break
                ;;
            *)
                print_error "Please enter 1 or 2"
                ;;
        esac
    done

    echo ""
}

################################################################################
# Step 3: Docker Installation
################################################################################

install_docker() {
    print_header "Step 3: Docker Installation"

    # Check if Docker is already installed
    if command -v docker &>/dev/null; then
        DOCKER_VERSION=$(docker --version)
        print_success "Docker already installed: $DOCKER_VERSION"

        # Check if user is in docker group
        if groups "$ACTUAL_USER" | grep -q docker; then
            print_success "User '$ACTUAL_USER' is in docker group"
        else
            print_warning "User '$ACTUAL_USER' not in docker group"
            if ask_yes_no "Add user to docker group?" "y"; then
                usermod -aG docker "$ACTUAL_USER"
                print_success "Added '$ACTUAL_USER' to docker group"
                print_info "  To activate immediately: newgrp docker"
                print_info "  Or log out and back in for permanent activation"
            fi
        fi
        echo ""
        return 0
    fi

    print_info "Docker not found. Installing Docker..."
    echo ""

    if ! ask_yes_no "Install Docker now?" "y"; then
        print_error "Docker is required for the acquisition system"
        exit 1
    fi

    print_info "Installing Docker (this may take a few minutes)..."

    # Update package index
    apt-get update -qq

    # Install prerequisites
    apt-get install -y -qq ca-certificates curl gnupg

    # Add Docker's official GPG key
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg

    # Set up Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker Engine
    apt-get update -qq
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Add user to docker group
    usermod -aG docker "$ACTUAL_USER"

    # Start Docker service
    systemctl start docker
    systemctl enable docker

    if command -v docker &>/dev/null; then
        print_success "Docker installed successfully"
        print_success "User '$ACTUAL_USER' added to docker group"
        print_info "  To activate immediately: newgrp docker"
        print_info "  Or log out and back in for permanent activation"
    else
        print_error "Docker installation failed"
        exit 1
    fi

    echo ""
}

################################################################################
# Step 4: Network Interface Detection
################################################################################

detect_network_interface() {
    print_header "Step 4: Network Interface Detection"

    print_info "Detecting network interfaces..."
    echo ""

    # Get list of physical ethernet interfaces (exclude virtual, loopback, etc.)
    interfaces=($(ip link show | grep -E '^[0-9]+:' | awk '{print $2}' | sed 's/:$//' | grep -E '^(en|eth)'))

    if [ ${#interfaces[@]} -eq 0 ]; then
        print_error "No ethernet interfaces found"
        print_info "Please ensure your network adapter is connected"
        exit 1
    fi

    print_info "Available ethernet interfaces:"
    echo ""

    for i in "${!interfaces[@]}"; do
        iface="${interfaces[$i]}"
        # Get interface details
        mac=$(ip link show "$iface" | grep -oP 'link/ether \K[^ ]+' || echo "unknown")
        status=$(ip link show "$iface" | grep -oP 'state \K\w+' || echo "unknown")

        echo "  [$((i+1))] $iface"
        echo "      MAC: $mac"
        echo "      Status: $status"
        echo ""
    done

    # Auto-select if only one interface
    if [ ${#interfaces[@]} -eq 1 ]; then
        NETWORK_INTERFACE="${interfaces[0]}"
        print_success "Auto-selected: $NETWORK_INTERFACE (only interface found)"
    else
        # Let user choose
        while true; do
            read -p "Select network interface [1-${#interfaces[@]}]: " iface_choice
            if [[ "$iface_choice" =~ ^[0-9]+$ ]] && [ "$iface_choice" -ge 1 ] && [ "$iface_choice" -le ${#interfaces[@]} ]; then
                NETWORK_INTERFACE="${interfaces[$((iface_choice-1))]}"
                print_success "Selected: $NETWORK_INTERFACE"
                break
            else
                print_error "Please enter a number between 1 and ${#interfaces[@]}"
            fi
        done
    fi

    echo ""
}

################################################################################
# Step 5: DHCP Server Setup (Laptop Mode Only)
################################################################################

setup_dhcp_server() {
    if [ "$DEPLOYMENT_MODE" != "laptop" ]; then
        print_info "Skipping DHCP setup (network mode selected)"
        return 0
    fi

    print_header "Step 5: DHCP Server Setup"

    # Check if isc-dhcp-server is installed
    if ! dpkg -l | grep -q isc-dhcp-server; then
        print_info "Installing isc-dhcp-server..."
        apt-get update -qq
        apt-get install -y isc-dhcp-server
        print_success "isc-dhcp-server installed"
    else
        print_success "isc-dhcp-server already installed"
    fi

    echo ""
    print_info "Detecting MAC address for network interface..."

    # Auto-detect MAC address of the selected network interface
    INTERFACE_MAC=$(ip link show "$NETWORK_INTERFACE" | grep -oP 'link/ether \K[^ ]+')

    if [ -n "$INTERFACE_MAC" ]; then
        print_success "Detected MAC address: $INTERFACE_MAC"
    else
        print_error "Could not detect MAC address for $NETWORK_INTERFACE"
        exit 1
    fi

    echo ""
    print_info "Configuring DHCP server..."

    # Configure /etc/dhcp/dhcpd.conf
    if [ -f /etc/dhcp/dhcpd.conf ]; then
        print_warning "DHCP configuration file already exists: /etc/dhcp/dhcpd.conf"

        # Check what's already configured
        local has_lease_time=$(grep -q "^default-lease-time" /etc/dhcp/dhcpd.conf && echo "yes" || echo "no")
        local has_subnet=$(grep -q "subnet 192.168.1.0" /etc/dhcp/dhcpd.conf && echo "yes" || echo "no")

        if [ "$has_lease_time" = "yes" ] && [ "$has_subnet" = "yes" ]; then
            # Config has required settings - check if MAC address matches
            local current_mac=""
            if grep -q "host laptop-interface" /etc/dhcp/dhcpd.conf; then
                current_mac=$(grep -A2 "host laptop-interface" /etc/dhcp/dhcpd.conf | grep "hardware ethernet" | awk '{print $3}' | tr -d ';')
            fi

            if [ -n "$current_mac" ] && [ "$current_mac" = "$INTERFACE_MAC" ]; then
                # Everything matches - no changes needed
                print_success "DHCP config already correct (lease times, subnet, and MAC address $INTERFACE_MAC)"
            elif [ -n "$current_mac" ]; then
                # MAC address mismatch
                print_warning "DHCP config has different MAC address"
                print_info "Configured MAC: $current_mac"
                print_info "Detected MAC:   $INTERFACE_MAC"
                echo ""

                if ask_yes_no "Update DHCP config with new MAC address?"; then
                    # Create backup
                    local backup_file="/etc/dhcp/dhcpd.conf.backup.$(date +%Y%m%d_%H%M%S)"
                    cp /etc/dhcp/dhcpd.conf "$backup_file"
                    print_info "Backup created: $backup_file"

                    # Recreate with new MAC
                    cat > /etc/dhcp/dhcpd.conf <<EOF
default-lease-time 600;
max-lease-time 7200;

subnet 192.168.1.0 netmask 255.255.255.0 {
    range 192.168.1.10 192.168.1.100;
    option domain-name-servers 8.8.8.8, 8.8.4.4;
    option domain-name "acquisition";
    option routers 192.168.1.1;
    option broadcast-address 192.168.1.255;

    host laptop-interface {
        hardware ethernet $INTERFACE_MAC;
        fixed-address 192.168.1.1;
    }
}
EOF
                    print_success "Updated /etc/dhcp/dhcpd.conf with new MAC address"
                else
                    print_info "Keeping existing DHCP configuration"
                fi
            else
                # No laptop-interface host block found - this is unusual
                print_warning "DHCP config missing laptop-interface host block"
                echo ""

                if ask_yes_no "Add laptop-interface host block with MAC $INTERFACE_MAC?"; then
                    # Create backup
                    local backup_file="/etc/dhcp/dhcpd.conf.backup.$(date +%Y%m%d_%H%M%S)"
                    cp /etc/dhcp/dhcpd.conf "$backup_file"
                    print_info "Backup created: $backup_file"

                    # Append host block to existing subnet
                    # This is a simple append - more complex logic would modify the subnet block in place
                    sed -i "/subnet 192.168.1.0/,/^}/ { /^}/i\\
\\    host laptop-interface {\\
\\        hardware ethernet $INTERFACE_MAC;\\
\\        fixed-address 192.168.1.1;\\
\\    }
}" /etc/dhcp/dhcpd.conf
                    print_success "Added laptop-interface host block"
                else
                    print_info "Keeping existing DHCP configuration"
                fi
            fi
        else
            # Config exists but is missing required settings - update it
            print_info "DHCP config exists but missing required settings. Updating..."

            # Create backup
            local backup_file="/etc/dhcp/dhcpd.conf.backup.$(date +%Y%m%d_%H%M%S)"
            cp /etc/dhcp/dhcpd.conf "$backup_file"
            print_info "Backup created: $backup_file"

            # Add missing lease times if needed
            if [ "$has_lease_time" = "no" ]; then
                sed -i '1i default-lease-time 600;\nmax-lease-time 7200;\n' /etc/dhcp/dhcpd.conf
                print_success "Added lease time settings"
            fi

            # Add subnet block if needed
            if [ "$has_subnet" = "no" ]; then
                cat >> /etc/dhcp/dhcpd.conf <<EOF

subnet 192.168.1.0 netmask 255.255.255.0 {
    range 192.168.1.10 192.168.1.100;
    option domain-name-servers 8.8.8.8, 8.8.4.4;
    option domain-name "acquisition";
    option routers 192.168.1.1;
    option broadcast-address 192.168.1.255;

    host laptop-interface {
        hardware ethernet $INTERFACE_MAC;
        fixed-address 192.168.1.1;
    }
}
EOF
                print_success "Added subnet 192.168.1.0 configuration"
            fi
        fi
    else
        # No existing config - create new one
        cat > /etc/dhcp/dhcpd.conf <<EOF
default-lease-time 600;
max-lease-time 7200;

subnet 192.168.1.0 netmask 255.255.255.0 {
    range 192.168.1.10 192.168.1.100;
    option domain-name-servers 8.8.8.8, 8.8.4.4;
    option domain-name "acquisition";
    option routers 192.168.1.1;
    option broadcast-address 192.168.1.255;

    host laptop-interface {
        hardware ethernet $INTERFACE_MAC;
        fixed-address 192.168.1.1;
    }
}
EOF
        print_success "Created /etc/dhcp/dhcpd.conf"
    fi

    # Configure /etc/default/isc-dhcp-server
    if [ -f /etc/default/isc-dhcp-server ]; then
        if grep -q "^INTERFACESv4=" /etc/default/isc-dhcp-server; then
            # Update existing line
            local current_interface=$(grep "^INTERFACESv4=" /etc/default/isc-dhcp-server | cut -d'"' -f2)
            if [ "$current_interface" = "$NETWORK_INTERFACE" ]; then
                print_success "INTERFACESv4 already set to $NETWORK_INTERFACE"
            else
                print_info "Updating INTERFACESv4 from $current_interface to $NETWORK_INTERFACE"
                sed -i "s/^INTERFACESv4=.*/INTERFACESv4=\"$NETWORK_INTERFACE\"/" /etc/default/isc-dhcp-server
                print_success "Updated INTERFACESv4 in /etc/default/isc-dhcp-server"
            fi
        else
            # Append new line
            echo "INTERFACESv4=\"$NETWORK_INTERFACE\"" >> /etc/default/isc-dhcp-server
            print_success "Added INTERFACESv4 to /etc/default/isc-dhcp-server"
        fi
    else
        # Create file
        echo "INTERFACESv4=\"$NETWORK_INTERFACE\"" > /etc/default/isc-dhcp-server
        print_success "Created /etc/default/isc-dhcp-server"
    fi

    # Create or update NetworkManager DHCP-Server profile
    echo ""
    print_info "Configuring NetworkManager DHCP-Server profile..."

    if nmcli con show DHCP-Server &>/dev/null; then
        print_warning "NetworkManager DHCP-Server profile already exists"

        if ask_yes_no "Do you want to update the existing profile?"; then
            print_info "Updating existing DHCP-Server profile..."
            nmcli con modify DHCP-Server \
                ifname "$NETWORK_INTERFACE" \
                autoconnect no \
                ipv4.method manual \
                ipv4.addresses 192.168.1.1/24 \
                ipv4.gateway 192.168.1.1 \
                ipv4.dns "8.8.8.8,8.8.4.4"
            print_success "Updated DHCP-Server profile"
        else
            print_info "Keeping existing DHCP-Server profile"
        fi
    else
        nmcli con add type ethernet con-name DHCP-Server ifname "$NETWORK_INTERFACE" \
            autoconnect no \
            ipv4.method manual \
            ipv4.addresses 192.168.1.1/24 \
            ipv4.gateway 192.168.1.1 \
            ipv4.dns "8.8.8.8,8.8.4.4"
        print_success "Created NetworkManager DHCP-Server profile"
    fi

    echo ""
}

################################################################################
# Step 6: Directory Creation
################################################################################

create_directories() {
    print_header "Step 6: Directory Creation"

    print_info "The acquisition system requires directories for data storage and camera configs"
    echo ""

    # Get data directory
    DATA_VOLUME=$(ask_input "Data storage directory" "/data")

    if [ -d "$DATA_VOLUME" ]; then
        print_success "Directory exists: $DATA_VOLUME"
    else
        print_info "Creating directory: $DATA_VOLUME"
        mkdir -p "$DATA_VOLUME"
        chown "$ACTUAL_USER:$ACTUAL_USER" "$DATA_VOLUME"
        print_success "Created: $DATA_VOLUME"
    fi

    # Get camera configs directory
    CAMERA_CONFIGS=$(ask_input "Camera configs directory" "/camera_configs")

    if [ -d "$CAMERA_CONFIGS" ]; then
        print_success "Directory exists: $CAMERA_CONFIGS"
    else
        print_info "Creating directory: $CAMERA_CONFIGS"
        mkdir -p "$CAMERA_CONFIGS"
        chown "$ACTUAL_USER:$ACTUAL_USER" "$CAMERA_CONFIGS"
        print_success "Created: $CAMERA_CONFIGS"
    fi

    # Get DataJoint external directory
    DATAJOINT_EXTERNAL=$(ask_input "DataJoint external storage directory" "/mnt/datajoint_external")

    if [ -d "$DATAJOINT_EXTERNAL" ]; then
        print_success "Directory exists: $DATAJOINT_EXTERNAL"
    else
        if ask_yes_no "Create DataJoint external directory?" "y"; then
            mkdir -p "$DATAJOINT_EXTERNAL"
            chown "$ACTUAL_USER:$ACTUAL_USER" "$DATAJOINT_EXTERNAL"
            print_success "Created: $DATAJOINT_EXTERNAL"
        else
            print_warning "Skipped creating $DATAJOINT_EXTERNAL (you can create it later)"
        fi
    fi

    echo ""
}

################################################################################
# Step 7: Environment Configuration
################################################################################

create_env_file() {
    print_header "Step 7: Environment Configuration"

    if [ -f ".env" ]; then
        print_warning ".env file already exists"
        if ! ask_yes_no "Overwrite existing .env file?" "n"; then
            print_info "Keeping existing .env file"
            echo ""
            return 0
        fi
    fi

    print_info "Creating .env file from template..."
    echo ""

    # Get database credentials
    print_info "DataJoint database configuration:"
    DJ_USER=$(ask_input "  Database username" "root")
    DJ_PASS=$(ask_input "  Database password" "pose")
    DJ_HOST=$(ask_input "  Database host" "127.0.0.1")
    DJ_PORT=$(ask_input "  Database port" "3306")

    echo ""

    # Other settings
    REACT_APP_BASE_URL=$(ask_input "React app base URL" "localhost")
    DISK_SPACE_WARNING_THRESHOLD_GB=$(ask_input "Disk space warning threshold (GB)" "50")

    # Create .env file
    cat > .env <<EOF
DJ_USER=$DJ_USER
DJ_PASS=$DJ_PASS
DJ_HOST=$DJ_HOST
DJ_PORT=$DJ_PORT
NETWORK_INTERFACE=$NETWORK_INTERFACE
DEPLOYMENT_MODE=$DEPLOYMENT_MODE
REACT_APP_BASE_URL=$REACT_APP_BASE_URL
DATA_VOLUME=$DATA_VOLUME
CAMERA_CONFIGS=$CAMERA_CONFIGS
DATAJOINT_EXTERNAL=$DATAJOINT_EXTERNAL
DISK_SPACE_WARNING_THRESHOLD_GB=$DISK_SPACE_WARNING_THRESHOLD_GB
EOF

    # Set ownership to actual user
    chown "$ACTUAL_USER:$ACTUAL_USER" .env

    print_success ".env file created"
    echo ""
}

################################################################################
# Step 8: Apply Persistence Settings
################################################################################

apply_persistence() {
    print_header "Step 8: Persistent Network Settings"

    print_info "Making network settings (MTU, buffers, DHCP) persist across reboots"
    echo ""

    if ask_yes_no "Apply persistent settings now?" "y"; then
        # Run persistence script as actual user
        sudo -u "$ACTUAL_USER" ./scripts/acquisition/make_settings_persistent.sh
        print_success "Persistence settings applied"
    else
        print_warning "Skipped persistence setup"
        print_info "You can run it later with: ./scripts/acquisition/make_settings_persistent.sh"
    fi

    echo ""
}

################################################################################
# Step 9: Build Docker Image
################################################################################

build_docker_image() {
    print_header "Step 9: Docker Image Build"

    print_info "The acquisition system requires building a Docker image"
    print_warning "This process can take 10-20 minutes"
    echo ""

    if ask_yes_no "Build Docker image now?" "y"; then
        print_info "Building mocap Docker image..."
        print_info "(You can watch the progress below)"
        echo ""

        # Run as actual user
        sudo -u "$ACTUAL_USER" make build-mocap

        if [ $? -eq 0 ]; then
            print_success "Docker image built successfully"
        else
            print_error "Docker build failed"
            print_info "You can try building again later with: make build-mocap"
        fi
    else
        print_warning "Skipped Docker build"
        print_info "You'll need to build before first use: make build-mocap"
    fi

    echo ""
}

################################################################################
# Summary and Next Steps
################################################################################

show_summary() {
    print_header "Setup Complete!"

    print_success "Acquisition system setup finished"
    echo ""

    print_info "Summary of configuration:"
    echo ""
    print_info "  Deployment mode: $DEPLOYMENT_MODE"
    print_info "  Network interface: $NETWORK_INTERFACE"
    print_info "  Data directory: $DATA_VOLUME"
    print_info "  Camera configs: $CAMERA_CONFIGS"
    echo ""

    print_info "Next steps:"
    echo ""
    print_info "  1. Activate docker group (choose one):"
    print_info "     - Run 'newgrp docker' to activate in current shell"
    print_info "     - Or log out and log back in for permanent activation"
    print_info "  2. Create camera configuration files in: $CAMERA_CONFIGS"
    print_info "     See: docs/acquisition/example_config.md"

    if [ "$DEPLOYMENT_MODE" = "laptop" ]; then
        print_info "  3. Connect cameras to network switch"
        print_info "  4. Start acquisition: ./scripts/acquisition/start_acquisition.sh"
    else
        print_info "  3. Ensure cameras are connected to building network"
        print_info "  4. Start acquisition: ./scripts/acquisition/start_acquisition.sh"
    fi

    echo ""
    print_info "Documentation: docs/README.md"
    print_info "Diagnostics: ./scripts/acquisition/acquisition_diagnostics.sh"
    echo ""
}

################################################################################
# Main Execution
################################################################################

main() {
    clear
    echo ""
    echo -e "${BOLD}${CYAN}╔═══════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}${CYAN}║  MultiCameraTracking Acquisition Setup Wizard     ║${NC}"
    echo -e "${BOLD}${CYAN}╚═══════════════════════════════════════════════════╝${NC}"
    echo ""

    print_info "This wizard will guide you through setting up the acquisition system"
    print_info "Press Ctrl+C at any time to cancel"
    echo ""

    if ! ask_yes_no "Continue with setup?" "y"; then
        print_info "Setup cancelled"
        exit 0
    fi

    # Run all setup steps
    check_prerequisites
    choose_deployment_mode
    install_docker
    detect_network_interface
    setup_dhcp_server
    create_directories
    create_env_file
    apply_persistence
    build_docker_image
    show_summary
}

# Run main function
main

exit 0
