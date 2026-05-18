"""Rescue cameras stuck on link-local 169.254.x.x via PySpin ForceIP.

When DHCP misses a camera at boot it falls back to link-local, which leaves
``Init()`` failing with ``Spinnaker: Camera is on a wrong subnet. [-1015]``
while ``make health`` still reports the camera as reachable (GVCP discovery
is link-local broadcast and crosses subnets). This script broadcasts a
``ForceIP`` GVCP command to each affected camera by MAC, assigning a fresh
address in the target subnet. The change is volatile — on the next camera
power-cycle DHCP runs again — so this is a runtime recovery, not a
permanent fix.

Usage (inside the test container):
    docker compose run --rm --entrypoint python3 test \\
        /Mocap/scripts/acquisition/force_camera_ips.py

By default rescues every camera whose current IP is in 169.254.0.0/16 and
assigns it ``192.168.1.{200+offset}`` with mask 255.255.255.0. Override with
``--target-subnet`` / ``--start-octet`` if your lab uses different ranges.
"""

from __future__ import annotations

import argparse
import ipaddress
import sys
import time


def _ip_to_int(ip: str) -> int:
    return int(ipaddress.IPv4Address(ip))


def _int_to_ip(value: int) -> str:
    return str(ipaddress.IPv4Address(value))


def _mac_to_str(mac_int: int) -> str:
    bs = mac_int.to_bytes(6, "big")
    return ":".join(f"{b:02x}" for b in bs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-subnet",
        default="192.168.1.0/24",
        help="Subnet to ForceIP cameras onto (default: 192.168.1.0/24).",
    )
    parser.add_argument(
        "--start-octet",
        type=int,
        default=200,
        help=(
            "Last octet to start assigning from (default: 200). Picked above "
            "the DHCP pool so the forced IPs don't collide with leases."
        ),
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="Pause after each ForceIP for the camera to apply (default: 2s).",
    )
    args = parser.parse_args()

    try:
        import PySpin  # type: ignore[import-untyped]
    except ImportError:
        print(
            "PySpin not importable — run this inside the test container.",
            file=sys.stderr,
        )
        return 2

    subnet = ipaddress.IPv4Network(args.target_subnet)
    mask_int = _ip_to_int(str(subnet.netmask))
    gateway_int = 0  # no gateway

    system = PySpin.System.GetInstance()
    try:
        iface_list = system.GetInterfaces()
        offset = 0
        any_rescued = False

        for i in range(iface_list.GetSize()):
            iface = iface_list.GetByIndex(i)
            iface_nodemap = iface.GetTLNodeMap()
            cam_list = iface.GetCameras()
            try:
                for j in range(cam_list.GetSize()):
                    cam = cam_list.GetByIndex(j)
                    tl = cam.GetTLDeviceNodeMap()

                    serial = PySpin.CStringPtr(tl.GetNode("DeviceSerialNumber"))
                    if not (PySpin.IsAvailable(serial) and PySpin.IsReadable(serial)):
                        continue
                    serial_str = serial.GetValue()

                    cur_ip_node = PySpin.CIntegerPtr(tl.GetNode("GevDeviceIPAddress"))
                    if not (
                        PySpin.IsAvailable(cur_ip_node)
                        and PySpin.IsReadable(cur_ip_node)
                    ):
                        continue
                    cur_ip = _int_to_ip(cur_ip_node.GetValue())

                    if not cur_ip.startswith("169.254."):
                        print(
                            f"{serial_str}: {cur_ip} — already on a routable subnet, skipping."
                        )
                        continue

                    mac_node = PySpin.CIntegerPtr(tl.GetNode("GevDeviceMACAddress"))
                    mac_int = mac_node.GetValue()

                    new_ip_int = (
                        _ip_to_int(str(subnet.network_address))
                        + args.start_octet
                        + offset
                    )
                    offset += 1
                    new_ip = _int_to_ip(new_ip_int)

                    print(
                        f"{serial_str} (MAC {_mac_to_str(mac_int)}): "
                        f"{cur_ip} → {new_ip} (mask {subnet.netmask})"
                    )

                    PySpin.CIntegerPtr(
                        iface_nodemap.GetNode("GevDeviceMACAddress")
                    ).SetValue(mac_int)
                    PySpin.CIntegerPtr(
                        iface_nodemap.GetNode("GevDeviceForceIPAddress")
                    ).SetValue(new_ip_int)
                    PySpin.CIntegerPtr(
                        iface_nodemap.GetNode("GevDeviceForceSubnetMask")
                    ).SetValue(mask_int)
                    PySpin.CIntegerPtr(
                        iface_nodemap.GetNode("GevDeviceForceGateway")
                    ).SetValue(gateway_int)
                    PySpin.CCommandPtr(
                        iface_nodemap.GetNode("GevDeviceForceIP")
                    ).Execute()

                    time.sleep(args.settle_seconds)
                    any_rescued = True
            finally:
                cam_list.Clear()

        iface_list.Clear()

        if not any_rescued:
            print("No cameras on 169.254.x.x — nothing to do.")
            return 0
        return 0
    finally:
        system.ReleaseInstance()


if __name__ == "__main__":
    raise SystemExit(main())
