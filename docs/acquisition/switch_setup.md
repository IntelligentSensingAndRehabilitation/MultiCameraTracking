# Switch Setup

The acquisition system uses NETGEAR MS510TXPP and MS510TXUP PoE switches
to power and connect the cameras.  Each switch needs a one-time SNMP
configuration so the acquisition tools can discover it and control PoE
ports programmatically (for camera power cycling during diagnostics and
remediation).

## Prerequisites

The laptop must already be set up via the setup wizard
(`setup_acquisition_system.sh`), which installs `lldpd` and `snmp`
automatically.  See [Automated Setup](automated_setup.md).

## Find the switch's IP

Connect the switch to the laptop's 10 GbE adapter and wait about 30
seconds for the switch to get a DHCP lease and for LLDP neighbors to
appear.  Then run:

```bash
python3 scripts/acquisition/switch_discover.py <interface>
```

The output shows the switch model, MAC address, and IP.  Open that IP
in a browser to access the switch's web management interface.

## One-time SNMP setup

The menu layout differs between the two switch models, but the
configuration values are the same.

### MS510TXPP

1. Log into the web interface at the switch's IP.  If this is a new
   switch the default password is `password` — change it to the lab's
   standard computer password on first login.
2. Click the **SNMP** tab in the top navigation.
3. Under community configuration, fill in:
   - Management Station IP: `192.168.1.0`
   - Management Station IP Mask: `255.255.255.0`
   - Community String: `isr-switch`
   - Access Mode: **Read/Write**
   - Status: **Enable**
4. Click **Add** (or **Apply**).

### MS510TXUP

1. Log into the web interface at the switch's IP.  If this is a new
   switch the default password is `password` — change it to the lab's
   standard computer password on first login.
2. Open the menu and navigate to **System > Protocols**.
3. Under community configuration, fill in:
   - Management Station IP: `192.168.1.0`
   - Management Station IP Mask: `255.255.255.0`
   - Community String: `isr-switch`
   - Access Mode: **Read/Write**
   - Status: **Enable**
4. Click **Add** (or **Apply**).

The management station IP and mask define which devices on the network
are allowed to send SNMP commands to the switch.  `192.168.1.0/24`
allows any device on the camera subnet.

## Verify SNMP access

From the laptop, confirm the switch responds to SNMP:

```bash
snmpget -v2c -c isr-switch <switch-ip> 1.3.6.1.2.1.1.1.0
```

This should return the switch's system description string.  If it times
out, double-check that the community was saved and that the access mode
is Read/Write.

## Verify PoE control

Check the PoE port status:

```bash
python3 scripts/acquisition/poe_cycle.py --status -i <interface> --community isr-switch
```

Power cycle all ports (cameras will reboot):

```bash
python3 scripts/acquisition/poe_cycle.py --cycle-all -i <interface> --community isr-switch
```

## Environment variables

To avoid passing `--interface` and `--community` every time, set these
in the shell or in the `.env` file:

```bash
export NETWORK_INTERFACE=enp37s0
export SWITCH_SNMP_COMMUNITY=isr-switch
```

Then the commands simplify to:

```bash
python3 scripts/acquisition/switch_discover.py
python3 scripts/acquisition/poe_cycle.py --status
python3 scripts/acquisition/poe_cycle.py --cycle-all
```

## Notes

- The SNMP configuration persists on the switch across power cycles and
  firmware updates.  It only needs to be done once per switch.
- Swapping which switch is connected to a laptop does not require any
  laptop-side reconfiguration.  The tools discover the switch
  automatically via LLDP and the DHCP lease table.
- With two daisy-chained switches, only the directly connected switch
  is visible via LLDP.  Use `--switch <ip>` to target the second
  switch, finding its IP from the DHCP lease table or the first
  switch's web interface.
