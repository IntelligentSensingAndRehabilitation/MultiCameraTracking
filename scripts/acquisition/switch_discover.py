#!/usr/bin/env python3
"""Discover network switches on the camera link via LLDP.

LLDP (IEEE 802.1AB) is enabled by default on NETGEAR MS510TXPP and
MS510TXUP switches regardless of firmware version.  Unlike Netgear's
proprietary NSDP, which was removed in firmware 1.1.0.8+, LLDP is an
IEEE standard that will not disappear in a firmware update.

Prerequisites::

    sudo apt install lldpd
    sudo systemctl start lldpd

Usage::

    python3 switch_discover.py [interface]
    python3 switch_discover.py enp37s0

LLDP neighbors take ~30 seconds to appear after lldpd starts.  With
daisy-chained switches only the directly connected switch is visible
via LLDP; use the DHCP lease table or --switch on poe_cycle.py to
reach the second switch.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

DEFAULT_LEASE_FILE = Path("/var/lib/dhcp/dhcpd.leases")


def _parse_active_leases(lease_file=DEFAULT_LEASE_FILE):
    """Return a dict mapping MAC address to leased IP from the ISC lease file.

    Only returns leases with ``binding state active``.
    """
    mac_to_ip = {}
    if not lease_file.exists():
        return mac_to_ip

    try:
        text = lease_file.read_text()
    except OSError:
        return mac_to_ip

    current_ip = None
    current_mac = None
    active = False
    for line in text.splitlines():
        stripped = line.strip()
        m = re.match(r"lease\s+([\d.]+)\s*\{", stripped)
        if m:
            current_ip = m.group(1)
            current_mac = None
            active = False
            continue
        if stripped == "}":
            if current_ip and current_mac and active:
                mac_to_ip[current_mac] = current_ip
            current_ip = None
            continue
        if current_ip is None:
            continue
        m = re.match(r"hardware ethernet\s+([\da-fA-F:]+)\s*;", stripped)
        if m:
            current_mac = m.group(1).lower()
        elif stripped == "binding state active;":
            active = True

    return mac_to_ip


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
    """Return LLDP neighbors, optionally filtered to one interface.

    When a neighbor advertises a ChassisID MAC but no MgmtIP, the DHCP
    lease table is checked to resolve the MAC to an IP.
    """
    try:
        result = subprocess.run(
            ["lldpcli", "show", "neighbors"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        print(
            "lldpcli not found. Install with: sudo apt install lldpd",
            file=sys.stderr,
        )
        return []
    except subprocess.TimeoutExpired:
        print("lldpcli timed out.", file=sys.stderr)
        return []

    if result.returncode != 0:
        print(f"lldpcli failed: {result.stderr.strip()}", file=sys.stderr)
        return []

    neighbors = parse_lldp_neighbors(result.stdout)
    if interface:
        neighbors = [n for n in neighbors if n.get("interface") == interface]

    # Fill in missing mgmt_ip from the DHCP lease table.
    leases = None
    for n in neighbors:
        if "mgmt_ip" not in n and "chassis_id" in n:
            if leases is None:
                leases = _parse_active_leases(lease_file)
            mac = _extract_mac(n["chassis_id"])
            ip = leases.get(mac)
            if ip:
                n["mgmt_ip"] = ip
                n["mgmt_ip_source"] = "dhcp"

    return neighbors


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
        print(f"No LLDP neighbors found on {interface}.")
        print("Ensure lldpd is running: sudo systemctl start lldpd")
        print("LLDP neighbors appear after ~30 seconds.")
        sys.exit(1)

    print(f"LLDP neighbors on {interface}:\n")
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
        print()


if __name__ == "__main__":
    main()
