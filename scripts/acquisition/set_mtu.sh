#!/bin/bash
# Source the environment variables from the .env file at the repo root.
# Works regardless of the caller's current working directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [ -f "$ENV_FILE" ]; then
  export $(grep -v '^#' "$ENV_FILE" | xargs)
else
  echo "Error: .env file not found at $ENV_FILE" >&2
  echo "Create it from .env.template first." >&2
  exit 1
fi

if [ -z "$NETWORK_INTERFACE" ]; then
  echo "Error: NETWORK_INTERFACE not set in $ENV_FILE" >&2
  exit 1
fi

sudo ip link set "${NETWORK_INTERFACE}" mtu 9000

sysctl -w net.core.rmem_max=10000000
sysctl -w net.core.rmem_default=10000000
