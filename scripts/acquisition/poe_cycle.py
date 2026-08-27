#!/usr/bin/env python3
"""Power cycle PoE ports on NETGEAR switches via SNMP.

Discovers switches on the camera network via LLDP and uses the standard
POWER-ETHERNET-MIB (RFC 3621) to disable and re-enable PoE on specified
ports.  Works on both MS510TXPP and MS510TXUP.

Prerequisites::

    sudo apt install lldpd snmp
    sudo systemctl start lldpd

Each switch needs a one-time SNMP community setup via the web GUI:
System > SNMP > SNMPv1/v2 > Community Configuration, access Read/Write.

Environment variables::

    NETWORK_INTERFACE         Camera network interface (e.g. enp37s0)
    SWITCH_SNMP_COMMUNITY     SNMP community string

Usage::

    python3 poe_cycle.py --discover                          # find switches
    python3 poe_cycle.py --status                            # PoE state
    python3 poe_cycle.py --cycle-all                         # all ports, all switches
    python3 poe_cycle.py --cycle-port 3                      # single port
    python3 poe_cycle.py --cycle-port 3 --switch 192.168.1.77
    python3 poe_cycle.py --cycle-all --delay 10              # 10s off
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

POE_ADMIN_OID = "1.3.6.1.2.1.105.1.1.1.3"
POE_PORTS = range(1, 9)
DEFAULT_DELAY = 5
DEFAULT_LEASE_FILE = Path("/var/lib/dhcp/dhcpd.leases")


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# LLDP discovery (duplicated from switch_discover.py so this script is
# self-contained — both run without a venv or project install).
# ---------------------------------------------------------------------------


def _parse_lldp_neighbors(text):
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
    return neighbors


def _parse_active_leases(lease_file=DEFAULT_LEASE_FILE):
    """Map MAC to IP from active ISC DHCP leases."""
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
    """Extract bare MAC from LLDP ChassisID (e.g. ``mac aa:bb:...``)."""
    parts = chassis_id.split()
    return (parts[-1] if parts else chassis_id).lower()


def discover_switches(interface):
    """Find switches via LLDP, with DHCP lease fallback for the IP."""
    try:
        result = subprocess.run(
            ["lldpcli", "show", "neighbors"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        die("lldpcli not found. Install with: sudo apt install lldpd")
    except subprocess.TimeoutExpired:
        die("lldpcli timed out")

    if result.returncode != 0:
        die(f"lldpcli failed: {result.stderr.strip()}")

    neighbors = _parse_lldp_neighbors(result.stdout)
    if interface:
        neighbors = [n for n in neighbors if n.get("interface") == interface]

    leases = None
    for n in neighbors:
        if "mgmt_ip" not in n and "chassis_id" in n:
            if leases is None:
                leases = _parse_active_leases()
            mac = _extract_mac(n["chassis_id"])
            ip = leases.get(mac)
            if ip:
                n["mgmt_ip"] = ip

    return [n for n in neighbors if "mgmt_ip" in n]


# ---------------------------------------------------------------------------
# SNMP helpers (shell out to net-snmp tools — no pysnmp dependency).
# ---------------------------------------------------------------------------


def snmp_get(host, oid, community):
    try:
        result = subprocess.run(
            ["snmpget", "-v2c", "-c", community, host, oid],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        die("snmpget not found. Install with: sudo apt install snmp")
    if result.returncode != 0:
        return None
    m = re.search(r":\s*(\S+)\s*$", result.stdout.strip())
    return m.group(1) if m else None


def snmp_set(host, oid, value, community):
    try:
        result = subprocess.run(
            ["snmpset", "-v2c", "-c", community, host, oid, "i", str(value)],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        die("snmpset not found. Install with: sudo apt install snmp")
    if result.returncode != 0:
        print(f"  snmpset failed: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------------------
# PoE operations
# ---------------------------------------------------------------------------


def get_poe_status(host, community):
    status = {}
    for port in POE_PORTS:
        val = snmp_get(host, f"{POE_ADMIN_OID}.1.{port}", community)
        if val is not None:
            status[port] = "enabled" if val == "1" else "disabled"
    return status


def cycle_ports(host, ports, community, delay):
    port_label = ", ".join(str(p) for p in ports)

    print(f"  Disabling PoE on port(s) {port_label} ...")
    for port in ports:
        if not snmp_set(host, f"{POE_ADMIN_OID}.1.{port}", 2, community):
            print(f"  Failed to disable port {port}", file=sys.stderr)
            return False

    print(f"  Waiting {delay}s ...")
    time.sleep(delay)

    print(f"  Re-enabling PoE on port(s) {port_label} ...")
    for port in ports:
        if not snmp_set(host, f"{POE_ADMIN_OID}.1.{port}", 1, community):
            print(f"  Failed to re-enable port {port}", file=sys.stderr)
            return False

    print("  Done.")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_switches(args):
    """Return a list of switch dicts from --switch or LLDP discovery."""
    if args.switch:
        return [{"mgmt_ip": args.switch, "description": args.switch}]

    if not args.interface:
        die("specify --interface or set NETWORK_INTERFACE")

    switches = discover_switches(args.interface)
    if not switches:
        die(
            f"no switches found via LLDP on {args.interface}. "
            "Ensure lldpd is running (sudo systemctl start lldpd) "
            "and wait ~30s for neighbors to appear. For a "
            "daisy-chained second switch, use --switch <ip>."
        )
    return switches


def main():
    parser = argparse.ArgumentParser(
        description="Power cycle PoE ports on NETGEAR switches via SNMP.",
    )
    parser.add_argument(
        "-i",
        "--interface",
        default=os.environ.get("NETWORK_INTERFACE"),
        help="Network interface (default: $NETWORK_INTERFACE)",
    )
    parser.add_argument(
        "--community",
        default=os.environ.get("SWITCH_SNMP_COMMUNITY"),
        help="SNMP community string (default: $SWITCH_SNMP_COMMUNITY)",
    )
    parser.add_argument(
        "--switch",
        help="Target switch IP (default: auto-discover via LLDP)",
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=int,
        default=DEFAULT_DELAY,
        help=f"Seconds to keep PoE off (default: {DEFAULT_DELAY})",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--discover",
        action="store_true",
        help="Discover switches on the network",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Show PoE port status",
    )
    group.add_argument(
        "--cycle-all",
        action="store_true",
        help="Power cycle all PoE ports",
    )
    group.add_argument(
        "--cycle-port",
        type=int,
        metavar="PORT",
        help="Power cycle a specific port (1-8)",
    )

    args = parser.parse_args()

    if args.cycle_port is not None and args.cycle_port not in POE_PORTS:
        die(f"port must be 1-8, got {args.cycle_port}")

    if not args.discover and not args.community:
        die(
            "SNMP community not set. Pass --community or set "
            "SWITCH_SNMP_COMMUNITY in the environment."
        )

    switches = resolve_switches(args)

    if args.discover:
        print(f"Switches found via LLDP on {args.interface}:\n")
        for s in switches:
            print(f"  {s.get('description', 'unknown')}")
            if "mgmt_ip" in s:
                print(f"    IP:   {s['mgmt_ip']}")
            if "chassis_id" in s:
                print(f"    MAC:  {s['chassis_id']}")
            print()
        return

    if args.status:
        for s in switches:
            ip = s["mgmt_ip"]
            print(f"\n{s.get('description', ip)} ({ip}):")
            status = get_poe_status(ip, args.community)
            if not status:
                print("  Could not read PoE status (SNMP unreachable?)")
                continue
            for port, state in sorted(status.items()):
                print(f"  Port {port}: {state}")
        return

    if args.cycle_port is not None and len(switches) > 1:
        die(
            "multiple switches found — use --switch to specify which "
            "one to cycle a single port on"
        )

    ports = list(POE_PORTS) if args.cycle_all else [args.cycle_port]

    for s in switches:
        ip = s["mgmt_ip"]
        print(f"\n{s.get('description', ip)} ({ip}):")
        cycle_ports(ip, ports, args.community, args.delay)


if __name__ == "__main__":
    main()
