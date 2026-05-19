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

from multi_camera.acquisition.flir_recording_api import (
    _ipv4_to_int,
    _enumerate_camera_ips,
    _force_one_camera_ip,
)
from multi_camera.acquisition.health import _get_host_interface_network


def _int_to_ip(value: int) -> str:
    return str(ipaddress.IPv4Address(value))


def _mac_to_str(mac_int: int) -> str:
    bs = mac_int.to_bytes(6, "big")
    return ":".join(f"{b:02x}" for b in bs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        default=os.environ.get("NETWORK_INTERFACE"),
        help=(
            "Host NIC to auto-detect the target subnet from. "
            "Defaults to $NETWORK_INTERFACE; required if neither is set."
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

    if not args.interface and not args.target_subnet:
        print(
            "No network interface specified. Pass --interface, set "
            "$NETWORK_INTERFACE, or pass --target-subnet to override the "
            "auto-detect path.",
            file=sys.stderr,
        )
        return 5

    # Gate to laptop deployment mode. In network mode the operator owns
    # the upstream DHCP fix; this CLI's volatile ForceIP rescue would
    # mask the real problem and require re-running on every camera
    # power-cycle. Lift this gate once we've designed a deliberate
    # network-mode ForceIP policy (target-subnet validation, lease-pool
    # collision avoidance against the upstream DHCP server, etc.).
    deployment_mode = os.environ.get("DEPLOYMENT_MODE", "laptop").strip().lower()
    if deployment_mode != "laptop":
        print(
            f"Force IP is only available in laptop deployment mode "
            f"(current DEPLOYMENT_MODE={deployment_mode!r}). In network "
            f"mode, fix the upstream DHCP server so cameras receive leases "
            f"on the correct subnet on their own.",
            file=sys.stderr,
        )
        return 4

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

    mask_int = _ipv4_to_int(str(subnet.netmask))
    gateway_int = 0
    network_base = _ipv4_to_int(str(subnet.network_address))
    print(f"Target subnet: {subnet} (mask {subnet.netmask})")

    system = PySpin.System.GetInstance()
    try:
        in_use_ips, off_subnet = _enumerate_camera_ips(system, subnet)
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
            new_ip_int = _ipv4_to_int(new_ip)
            print(
                f"{serial} (MAC {_mac_to_str(mac_int)}): {cur_ip} → "
                f"{new_ip} (mask {subnet.netmask})"
            )

            if not _force_one_camera_ip(
                system, mac_int, new_ip_int, mask_int, gateway_int
            ):
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
            _, still_off = _enumerate_camera_ips(system, subnet)
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
