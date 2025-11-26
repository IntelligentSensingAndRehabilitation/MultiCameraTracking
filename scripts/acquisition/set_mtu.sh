#/bin/env sh
# Source the environment variables from the .env file
# Change the path to your .env file if necessary
ENV_FILE=".env"

if [ -f $ENV_FILE ]; then
  export $(grep -v '^#' $ENV_FILE | xargs)
fi

sudo ip link set ${NETWORK_INTERFACE} mtu 9000

sysctl -w net.core.rmem_max=10000000
sysctl -w net.core.rmem_default=10000000
