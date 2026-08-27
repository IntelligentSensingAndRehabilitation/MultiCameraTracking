# DHCP Server Setup

The acquisition system uses an ISC DHCP server running on the laptop to
assign IP addresses to the cameras.  This is only needed in laptop
(portable) mode; in network mode an upstream DHCP server is assumed.

## Automated setup

The setup wizard handles all DHCP configuration automatically:

```bash
sudo ./scripts/acquisition/setup_acquisition_system.sh
```

The wizard performs the following steps in laptop mode:

1. Installs `isc-dhcp-server` if not already present.
2. Detects the camera network interface (the 10 GbE adapter connected to
   the switch via Thunderbolt) and its MAC address.
3. Writes `/etc/dhcp/dhcpd.conf` with:
   - Subnet `192.168.1.0/24`, pool range `.10` to `.100`
   - Default lease 600 s, max lease 7200 s
4. Sets `INTERFACESv4` in `/etc/default/isc-dhcp-server` to the detected
   interface.
5. Creates (or updates) a NetworkManager connection profile named
   `DHCP-Server` bound to the detected interface with a static IP of
   `192.168.1.1/24` and `autoconnect yes`, so the profile activates
   whenever the interface comes up.

After the wizard finishes, run `make_settings_persistent.sh` (the wizard
calls this automatically) to enable `isc-dhcp-server` on boot and persist
MTU and buffer settings.

## What happens at startup

`start_acquisition.sh` (called by `make run`) activates the `DHCP-Server`
NetworkManager profile, verifies the interface has `192.168.1.1`, and
starts `isc-dhcp-server` if it is not already running.  If either step
fails, the script prints instructions for recovery.

## When to re-run the wizard

Re-run the wizard when the laptop changes.  The interface name (e.g.
`enp5s0`, `enp37s0`) is tied to the laptop's Thunderbolt adapter, so a
different laptop will have a different interface name.  The wizard
detects the new interface and updates all configuration files.

Swapping which switch is connected does not require any reconfiguration
on the laptop.

## Manual setup

If the wizard does not work for your environment, the individual steps
are listed below.  Replace `INTERFACE` with your camera network interface
name throughout.

### Find the interface name

Plug the Ethernet cable from the switch into the 10 GbE Thunderbolt
adapter, then run:

```bash
ip link show
```

Look for an interface named `enp<N>s0` (the number varies by laptop).
If multiple interfaces are present, unplug the cable and see which one
disappears.

### Install the DHCP server

```bash
sudo apt-get update
sudo apt-get install isc-dhcp-server
```

### Configure dhcpd

Write `/etc/dhcp/dhcpd.conf`:

```
default-lease-time 600;
max-lease-time 7200;

subnet 192.168.1.0 netmask 255.255.255.0 {
    range 192.168.1.10 192.168.1.100;
    option domain-name "acquisition";
    option broadcast-address 192.168.1.255;
}
```

The camera subnet is isolated (no upstream router), so `option routers`
and `option domain-name-servers` are omitted.

Set the DHCP listening interface in `/etc/default/isc-dhcp-server`:

```
INTERFACESv4="INTERFACE"
```

### Create the NetworkManager profile

```bash
nmcli con add type ethernet con-name DHCP-Server ifname INTERFACE \
    autoconnect yes ipv4.method manual ipv4.addresses 192.168.1.1/24
```

### Persist settings

```bash
sudo systemctl enable isc-dhcp-server
sudo ./scripts/acquisition/make_settings_persistent.sh
```

### Start the server

```bash
nmcli con up DHCP-Server
sudo systemctl start isc-dhcp-server
```
