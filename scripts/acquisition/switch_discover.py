#!/usr/bin/env python3
"""Discover network switches on the camera link.

Uses two complementary methods:

1. **LLDP** (IEEE 802.1AB) — finds the directly connected switch.
   Enabled by default on MS510TXPP and MS510TXUP.
2. **DHCP lease table** — finds any switch on the subnet by its
   vendor-class-identifier (``MS510TXPP``, ``MS510TXUP``, or
   ``Switch``).  This catches daisy-chained switches that are not
   directly visible via LLDP.

Prerequisites::

    sudo apt install lldpd
    sudo systemctl start lldpd

Usage::

    python3 switch_discover.py [interface]
    python3 switch_discover.py enp37s0
"""

import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_LEASE_FILE = Path("/var/lib/dhcp/dhcpd.leases")


SWITCH_VENDOR_CLASSES = {"MS510TXPP", "MS510TXUP", "Switch"}


def _parse_active_leases(lease_file=DEFAULT_LEASE_FILE):
    """Parse active leases from the ISC DHCP lease file.

    Returns a list of dicts with keys ``ip``, ``mac``, and optionally
    ``vendor_class``.  Only leases with ``binding state active`` are
    included.
    """
    leases = []
    if not lease_file.exists():
        return leases

    try:
        text = lease_file.read_text()
    except OSError:
        return leases

    current_ip = None
    current_mac = None
    current_vendor = None
    active = False
    for line in text.splitlines():
        stripped = line.strip()
        m = re.match(r"lease\s+([\d.]+)\s*\{", stripped)
        if m:
            current_ip = m.group(1)
            current_mac = None
            current_vendor = None
            active = False
            continue
        if stripped == "}":
            if current_ip and current_mac and active:
                entry = {"ip": current_ip, "mac": current_mac}
                if current_vendor:
                    entry["vendor_class"] = current_vendor
                leases.append(entry)
            current_ip = None
            continue
        if current_ip is None:
            continue
        m = re.match(r"hardware ethernet\s+([\da-fA-F:]+)\s*;", stripped)
        if m:
            current_mac = m.group(1).lower()
        elif stripped == "binding state active;":
            active = True
        else:
            m = re.match(
                r'set vendor-class-identifier\s*=\s*"([^"]+)"\s*;', stripped
            )
            if m:
                current_vendor = m.group(1)

    return leases


def _leases_by_mac(leases):
    """Index a lease list by MAC address."""
    return {entry["mac"]: entry["ip"] for entry in leases}


def _find_switch_leases(leases):
    """Return lease entries whose vendor class identifies them as switches."""
    return [
        entry
        for entry in leases
        if entry.get("vendor_class") in SWITCH_VENDOR_CLASSES
    ]


def _extract_mac(chassis_id):
    """Extract a bare MAC string from an LLDP ChassisID value.

    LLDP ChassisID may be ``mac 28:80:88:73:4c:70`` or just
    ``28:80:88:73:4c:70``.
    """
    parts = chassis_id.split()
    raw = parts[-1] if parts else chassis_id
    return raw.lower()


def parse_lldp_neighbors(text):
    """Parse ``lldpcli show neighbors`` text into a list of dicts."""
    neighbors = []
    current = None

    for line in text.splitlines():
        m = re.match(r"Interface:\s+(\S+),\s+via:\s+(\S+)", line)
        if m:
            current = {"interface": m.group(1), "via": m.group(2)}
            neighbors.append(current)
            continue
        if current is None:
            continue

        stripped = line.strip()
        if stripped.startswith("ChassisID:"):
            current["chassis_id"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("SysName:"):
            current["name"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("SysDescr:"):
            current["description"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("MgmtIP:"):
            current["mgmt_ip"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("PortID:"):
            current["port_id"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("PortDescr:"):
            current["port_descr"] = stripped.split(":", 1)[1].strip()

    return neighbors


def discover_switches(interface=None, lease_file=DEFAULT_LEASE_FILE):
    """Discover switches via LLDP and the DHCP lease table.

    LLDP finds the directly connected switch.  The lease table finds
    any switch on the subnet whose vendor-class-identifier marks it as
    a NETGEAR switch (including daisy-chained switches not visible via
    LLDP).  Results from both sources are merged and deduplicated by
    MAC address.
    """
    # -- LLDP ---------------------------------------------------------------
    lldp_neighbors = []
    try:
        result = subprocess.run(
            ["lldpcli", "show", "neighbors"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            lldp_neighbors = parse_lldp_neighbors(result.stdout)
            if interface:
                lldp_neighbors = [
                    n
                    for n in lldp_neighbors
                    if n.get("interface") == interface
                ]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # -- DHCP lease table ---------------------------------------------------
    all_leases = _parse_active_leases(lease_file)
    mac_to_ip = _leases_by_mac(all_leases)

    # Fill in missing mgmt_ip on LLDP neighbors.
    for n in lldp_neighbors:
        if "mgmt_ip" not in n and "chassis_id" in n:
            mac = _extract_mac(n["chassis_id"])
            ip = mac_to_ip.get(mac)
            if ip:
                n["mgmt_ip"] = ip
                n["mgmt_ip_source"] = "dhcp"

    # Build a set of MACs already covered by LLDP.
    seen_macs = set()
    for n in lldp_neighbors:
        if "chassis_id" in n:
            seen_macs.add(_extract_mac(n["chassis_id"]))

    # Add switches found only in the lease table (daisy-chained).
    switch_leases = _find_switch_leases(all_leases)
    for entry in switch_leases:
        if entry["mac"] not in seen_macs:
            lldp_neighbors.append(
                {
                    "mgmt_ip": entry["ip"],
                    "mgmt_ip_source": "dhcp",
                    "chassis_id": entry["mac"],
                    "description": entry.get("vendor_class", "unknown"),
                    "discovery": "dhcp-only",
                }
            )
            seen_macs.add(entry["mac"])

    return lldp_neighbors


def main():
    interface = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.environ.get("NETWORK_INTERFACE")
    )
    if not interface:
        print("Usage: switch_discover.py <interface>")
        print("  or set NETWORK_INTERFACE environment variable")
        sys.exit(1)

    neighbors = discover_switches(interface)

    if not neighbors:
        print(f"No switches found on {interface}.")
        print("Ensure lldpd is running: sudo systemctl start lldpd")
        print("LLDP neighbors appear after ~30 seconds.")
        print("Switches also need an active DHCP lease to be found")
        print("via the lease table.")
        sys.exit(1)

    print(f"Switches on {interface}:\n")
    for n in neighbors:
        print(f"  {n.get('description', 'unknown')}")
        if "mgmt_ip" in n:
            source = n.get("mgmt_ip_source", "lldp")
            print(f"    IP:        {n['mgmt_ip']}  (via {source})")
        if "chassis_id" in n:
            print(f"    MAC:       {n['chassis_id']}")
        if "port_id" in n:
            print(f"    Port:      {n['port_id']}")
        if "name" in n:
            print(f"    Name:      {n['name']}")
        if n.get("discovery") == "dhcp-only":
            print(f"    Note:      not directly connected (found via DHCP lease)")
        print()


if __name__ == "__main__":
    main()
