#/bin/env sh

sudo ip link set enp5s0 mtu 9000

sysctl -w net.core.rmem_max=10000000
sysctl -w net.core.rmem_default=10000000
