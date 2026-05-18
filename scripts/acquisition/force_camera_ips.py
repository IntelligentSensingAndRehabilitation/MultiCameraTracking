"""Rescue cameras stuck on a wrong subnet via Spinnaker ForceIP.

When DHCP misses a camera at boot it falls back to link-local 169.254.x.x,
which leaves ``Init()`` failing with ``Spinnaker: Camera is on a wrong
subnet. [-1015]`` while ``make health`` still reports the camera as
reachable (GVCP discovery is link-local broadcast and crosses subnets).
This script broadcasts a ``ForceIP`` GVCP command to each affected
camera, assigning a fresh IP within the host NIC's subnet. The change is
volatile — on the next camera power-cycle the camera renegotiates DHCP
or falls back to its persistent IP — so this is a runtime recovery, not
a permanent fix.

Usage (inside the test container):
    docker compose run --rm --entrypoint python3 test \\
        /Mocap/scripts/acquisition/force_camera_ips.py

By default the target subnet is auto-detected from the NIC named by
``$NETWORK_INTERFACE`` (or ``--interface``) via ``SIOCGIFNETMASK``. New
IPs are picked starting at ``--start-octet`` (default 240, chosen to sit
above typical DHCP pools that run .10–.100) and skip any IP already
held by another discovered camera so we never collide with a live
lease. Targeting works by re-enumerating the camera list right before
each ForceIP — that gives a fresh ``CameraPtr`` whose internal state
points at exactly one device — then MAC-verifying the handle before
writing ``GevDeviceForce*`` and executing ``GevDeviceForceIP`` on its
own ``GetTLDeviceNodeMap``. After the configured settle delay the
camera list is re-enumerated again to confirm the target actually
moved onto the new subnet; cameras that don't move (firmware refused,
camera offline, etc.) are reported so the operator can power-cycle
them instead.
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys
import time

from multi_camera.acquisition.health import _get_host_interface_network


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
        "--interface",
        default=os.environ.get("NETWORK_INTERFACE", "enp34s0"),
        help=(
            "Host NIC to auto-detect the target subnet from. "
            "Defaults to $NETWORK_INTERFACE."
        ),
    )
    parser.add_argument(
        "--target-subnet",
        default=None,
        help=(
            "Override the auto-detected subnet (e.g. 192.168.1.0/24). "
            "Use only if you know the host NIC's IP isn't on the same "
            "subnet the cameras should join."
        ),
    )
    parser.add_argument(
        "--start-octet",
        type=int,
        default=240,
        help=(
            "Last octet to start assigning from (default: 240). 240+ "
            "sits above typical isc-dhcp-server pools that run .10–.100, "
            "so forced IPs won't collide with a future DHCPACK."
        ),
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=5.0,
        help=(
            "Pause after each ForceIP for the camera to apply the new "
            "config (default: 5s). Shorter values can cause subsequent "
            "ForceIP calls to target the still-reconfiguring camera."
        ),
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

    if args.target_subnet:
        subnet = ipaddress.IPv4Network(args.target_subnet)
    else:
        subnet = _get_host_interface_network(args.interface)
        if subnet is None:
            print(
                f"Could not detect IPv4 subnet on interface {args.interface}. "
                f"Pass --target-subnet to override.",
                file=sys.stderr,
            )
            return 3

    mask_int = _ip_to_int(str(subnet.netmask))
    gateway_int = 0
    network_base = _ip_to_int(str(subnet.network_address))
    print(f"Target subnet: {subnet} (mask {subnet.netmask})")

    system = PySpin.System.GetInstance()
    try:

        def enumerate_state() -> tuple[set[str], list[tuple[str, int, str]]]:
            """Snapshot every camera's (serial, mac, current_ip) by walking
            every interface. Returns (in_use_ips_on_subnet, off_subnet_list).
            Re-runnable — used both up-front and again after each ForceIP to
            verify the camera actually moved.
            """
            in_use: set[str] = set()
            off: list[tuple[str, int, str]] = []
            iface_list = system.GetInterfaces()
            try:
                for iface_idx in range(iface_list.GetSize()):
                    iface = iface_list.GetByIndex(iface_idx)
                    cam_list = iface.GetCameras()
                    try:
                        for cam_idx in range(cam_list.GetSize()):
                            cam = cam_list.GetByIndex(cam_idx)
                            try:
                                tl = cam.GetTLDeviceNodeMap()
                                serial_node = PySpin.CStringPtr(
                                    tl.GetNode("DeviceSerialNumber")
                                )
                                if not (
                                    PySpin.IsAvailable(serial_node)
                                    and PySpin.IsReadable(serial_node)
                                ):
                                    continue
                                serial = serial_node.GetValue()
                                ip_node = PySpin.CIntegerPtr(
                                    tl.GetNode("GevDeviceIPAddress")
                                )
                                if not (
                                    PySpin.IsAvailable(ip_node)
                                    and PySpin.IsReadable(ip_node)
                                ):
                                    continue
                                cur_ip = _int_to_ip(ip_node.GetValue())
                                try:
                                    cur_ip_addr = ipaddress.IPv4Address(cur_ip)
                                except (ValueError, ipaddress.AddressValueError):
                                    continue
                                if cur_ip_addr in subnet:
                                    in_use.add(cur_ip)
                                    continue
                                mac_int = PySpin.CIntegerPtr(
                                    tl.GetNode("GevDeviceMACAddress")
                                ).GetValue()
                                off.append((serial, mac_int, cur_ip))
                            finally:
                                del cam
                    finally:
                        cam_list.Clear()
            finally:
                iface_list.Clear()
            return in_use, off

        def force_one(target_mac: int, new_ip_int: int) -> bool:
            """Re-enumerate, find the camera with the matching MAC, write
            ``GevDeviceForce*`` on its TLDeviceNodeMap, and Execute. The
            writes + Execute all happen inside one ``CameraPtr`` scope so
            Spinnaker can target through the handle. Returns True if a
            matching camera was found (the ForceIP packet was sent), False
            otherwise. Caller verifies after waiting.
            """
            iface_list = system.GetInterfaces()
            try:
                for iface_idx in range(iface_list.GetSize()):
                    iface = iface_list.GetByIndex(iface_idx)
                    cam_list = iface.GetCameras()
                    try:
                        for cam_idx in range(cam_list.GetSize()):
                            cam = cam_list.GetByIndex(cam_idx)
                            try:
                                tl = cam.GetTLDeviceNodeMap()
                                this_mac = PySpin.CIntegerPtr(
                                    tl.GetNode("GevDeviceMACAddress")
                                ).GetValue()
                                if this_mac != target_mac:
                                    continue
                                PySpin.CIntegerPtr(
                                    tl.GetNode("GevDeviceForceIPAddress")
                                ).SetValue(new_ip_int)
                                PySpin.CIntegerPtr(
                                    tl.GetNode("GevDeviceForceSubnetMask")
                                ).SetValue(mask_int)
                                PySpin.CIntegerPtr(
                                    tl.GetNode("GevDeviceForceGateway")
                                ).SetValue(gateway_int)
                                PySpin.CCommandPtr(
                                    tl.GetNode("GevDeviceForceIP")
                                ).Execute()
                                return True
                            finally:
                                del cam
                    finally:
                        cam_list.Clear()
            finally:
                iface_list.Clear()
            return False

        in_use_ips, off_subnet = enumerate_state()
        if not off_subnet:
            print("No cameras off-subnet — nothing to do.")
            return 0

        def next_free_ip() -> str:
            for octet in range(args.start_octet, 255):
                candidate = _int_to_ip(network_base + octet)
                if candidate not in in_use_ips:
                    in_use_ips.add(candidate)
                    return candidate
            raise RuntimeError(
                f"No free IPs in {subnet} starting at .{args.start_octet}."
            )

        failures: list[str] = []
        for serial, mac_int, cur_ip in off_subnet:
            new_ip = next_free_ip()
            new_ip_int = _ip_to_int(new_ip)
            print(
                f"{serial} (MAC {_mac_to_str(mac_int)}): {cur_ip} → "
                f"{new_ip} (mask {subnet.netmask})"
            )

            if not force_one(mac_int, new_ip_int):
                print(
                    f"  ! No camera with MAC {_mac_to_str(mac_int)} found on "
                    f"re-enumeration; skipping.",
                    file=sys.stderr,
                )
                failures.append(serial)
                continue

            time.sleep(args.settle_seconds)

            # Verify: if this MAC still shows up in the off-subnet list after
            # the settle, the ForceIP didn't stick — surface the camera so
            # the operator can power-cycle it.
            _, still_off = enumerate_state()
            if mac_int in {m for _, m, _ in still_off}:
                print(
                    f"  ! {serial} still off-subnet after {args.settle_seconds}s. "
                    f"The camera is likely ignoring GVCP due to a stale "
                    f"switch ARP entry or a dormant state. Try, in order:\n"
                    f"      1. sudo service isc-dhcp-server restart  "
                    f"(refreshes layer-2 traffic; usually wakes the camera)\n"
                    f"      2. make force-ip  (retry)\n"
                    f"      3. Power-cycle the camera as a last resort.",
                    file=sys.stderr,
                )
                failures.append(serial)
            else:
                print(f"  ✓ {serial} on {subnet}")

        if failures:
            print(f"\nUnresolved: {', '.join(failures)}", file=sys.stderr)
            return 1
        return 0
    finally:
        system.ReleaseInstance()


if __name__ == "__main__":
    raise SystemExit(main())
