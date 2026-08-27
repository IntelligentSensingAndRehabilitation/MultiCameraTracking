#!/usr/bin/env python3
"""Find NETGEAR smart switches on a link using NSDP.

NSDP (NETGEAR Switch Discovery Protocol) is the same broadcast protocol the
official NETGEAR Discovery Tool uses.  It works even when the switch is on a
different subnet than the host, because discovery happens over broadcast
rather than routed unicast.

Usage::

    sudo python3 nsdp_discover.py <interface>
    sudo python3 nsdp_discover.py enp37s0
"""

import socket
import struct
import sys
import time

CLIENT_PORT = 63321
SERVER_PORT = 63322

# TLV types we care about in the response.
TLV = {
    0x0001: "model",
    0x0003: "name",
    0x0004: "mac",
    0x0006: "ip",
    0x0007: "netmask",
    0x0008: "gateway",
    0x000B: "firmware",
    0x000C: "firmware2",
    0x000D: "dhcp",
}


def local_mac(ifname):
    """Read the interface MAC from sysfs."""
    with open(f"/sys/class/net/{ifname}/address") as fh:
        return bytes.fromhex(fh.read().strip().replace(":", ""))


def build_request(host_mac, seq):
    """Construct an NSDP read request asking for the interesting TLVs."""
    pkt = struct.pack(
        ">BBHI",
        0x01,        # version
        0x01,        # operation: read request
        0x0000,      # result code
        0x00000000,  # reserved
    )
    pkt += host_mac                 # host MAC
    pkt += b"\x00" * 6              # device MAC: zeros = "any device"
    pkt += b"\x00\x00"              # reserved
    pkt += struct.pack(">H", seq)   # sequence number
    pkt += b"NSDP"                  # signature
    pkt += b"\x00" * 4              # reserved

    # Request each TLV with zero length (a query, not a set).
    for tlv_type in sorted(TLV):
        pkt += struct.pack(">HH", tlv_type, 0)

    pkt += b"\xff\xff\x00\x00"      # end-of-message marker
    return pkt


def parse_response(data):
    """Walk the TLV list in a response and pull out readable fields."""
    if len(data) < 32 or data[24:28] != b"NSDP":
        return None

    fields = {}
    off = 32
    while off + 4 <= len(data):
        tlv_type, length = struct.unpack(">HH", data[off : off + 4])
        off += 4
        if tlv_type == 0xFFFF:
            break
        value = data[off : off + length]
        off += length

        name = TLV.get(tlv_type)
        if not name:
            continue

        if name in ("ip", "netmask", "gateway") and length == 4:
            fields[name] = socket.inet_ntoa(value)
        elif name == "mac" and length == 6:
            fields[name] = ":".join(f"{b:02x}" for b in value)
        elif name == "dhcp":
            fields[name] = "enabled" if value and value[-1] else "disabled"
        else:
            fields[name] = value.rstrip(b"\x00").decode("utf-8", "replace")

    return fields or None


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    ifname = sys.argv[1]
    host_mac = local_mac(ifname)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Bind to the interface so this works with no usable IP on the link.
    sock.setsockopt(
        socket.SOL_SOCKET, socket.SO_BINDTODEVICE, ifname.encode()
    )
    sock.bind(("", CLIENT_PORT))
    sock.settimeout(1.0)

    request = build_request(host_mac, seq=1)
    print(f"Broadcasting NSDP discovery on {ifname} ...\n")

    for _ in range(3):
        sock.sendto(request, ("255.255.255.255", SERVER_PORT))
        time.sleep(0.2)

    seen = set()
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            data, addr = sock.recvfrom(2048)
        except socket.timeout:
            continue

        info = parse_response(data)
        if not info:
            continue

        key = info.get("mac", addr[0])
        if key in seen:
            continue
        seen.add(key)

        print(f"  found: {info.get('model', 'unknown model')}")
        for field in (
            "name", "mac", "ip", "netmask", "gateway", "dhcp", "firmware",
        ):
            if field in info:
                print(f"    {field:<9} {info[field]}")
        print()

    if not seen:
        print("No switches responded. The switch may have NSDP disabled,")
        print("or the interface may not have carrier.")


if __name__ == "__main__":
    main()
