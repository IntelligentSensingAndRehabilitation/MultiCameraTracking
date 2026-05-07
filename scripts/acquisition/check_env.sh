#!/bin/bash
################################################################################
# check_env.sh
#
# Validate .env against .env.template. For each variable declared in the
# template that is missing or empty in .env, prompt the operator with a
# short description and persist the answer back to .env. Idempotent —
# variables that are already set are left alone.
#
# Usage:
#   ./scripts/acquisition/check_env.sh            Interactive validation
#   ./scripts/acquisition/check_env.sh --quiet    Exit non-zero if anything missing
################################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"
TEMPLATE_FILE="$REPO_ROOT/.env.template"

QUIET=false
[ "${1:-}" = "--quiet" ] && QUIET=true

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

print_header()  { echo ""; echo -e "${CYAN}=== $1 ===${NC}"; echo ""; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error()   { echo -e "${RED}✗${NC} $1"; }
print_info()    { echo "$1"; }

describe_var() {
    case "$1" in
        DJ_USER) echo "DataJoint database username" ;;
        DJ_PASS) echo "DataJoint database password" ;;
        DJ_HOST) echo "DataJoint database host (e.g. 127.0.0.1 for local)" ;;
        DJ_PORT) echo "DataJoint database port" ;;
        NETWORK_INTERFACE) echo "Camera network interface name (run 'ip link' to list); typical: enp5s0" ;;
        DEPLOYMENT_MODE) echo "'laptop' (built-in DHCP server) or 'network' (upstream DHCP)" ;;
        REACT_APP_BASE_URL) echo "Hostname/IP the GUI talks to; 'localhost' for same-machine browser" ;;
        DATA_VOLUME) echo "Recording storage path on host (e.g. /data)" ;;
        TEST_DATA_VOLUME) echo "Isolated test recording storage (used by 'make run-mocap-test')" ;;
        CAMERA_CONFIGS) echo "Path to camera config YAMLs (e.g. /camera_configs)" ;;
        DATAJOINT_EXTERNAL) echo "DataJoint external blob storage (e.g. /mnt/datajoint_external)" ;;
        DISK_SPACE_WARNING_THRESHOLD_GB) echo "Warn when free disk space falls below this many GB" ;;
        *) echo "" ;;
    esac
}

template_vars() {
    grep -E '^[A-Z_][A-Z_0-9]*=' "$TEMPLATE_FILE"
}

get_env_value() {
    local var="$1"
    [ -f "$ENV_FILE" ] || { echo ""; return; }
    grep -E "^${var}=" "$ENV_FILE" | tail -n1 | cut -d= -f2-
}

set_env_value() {
    local var="$1" value="$2"
    if [ -f "$ENV_FILE" ] && grep -qE "^${var}=" "$ENV_FILE"; then
        local tmp
        tmp="$(mktemp)"
        awk -v var="$var" -v val="$value" \
            'BEGIN { FS="="; OFS="=" }
             $1 == var { print var "=" val; next }
             { print }' "$ENV_FILE" > "$tmp"
        mv "$tmp" "$ENV_FILE"
    else
        printf '%s=%s\n' "$var" "$value" >> "$ENV_FILE"
    fi
}

print_header "Validating .env against .env.template"

if [ ! -f "$TEMPLATE_FILE" ]; then
    print_error ".env.template not found at $TEMPLATE_FILE"
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    if $QUIET; then
        print_error ".env not found at $ENV_FILE"
        exit 1
    fi
    print_warning ".env not found — will create from prompted values"
    touch "$ENV_FILE"
fi

missing_count=0
filled_vars=()

# Read template lines from fd 3 so the prompt's `read` keeps fd 0 (stdin).
while IFS= read -r -u 3 line; do
    var="${line%%=*}"
    template_default="${line#*=}"
    current=$(get_env_value "$var")
    if [ -z "$current" ]; then
        missing_count=$((missing_count + 1))
        if $QUIET; then
            print_error "$var is not set in .env"
            continue
        fi
        echo ""
        echo "$var"
        desc=$(describe_var "$var")
        [ -n "$desc" ] && echo "  $desc"
        if [ -n "$template_default" ]; then
            read -r -p "  value [$template_default]: " value
            value="${value:-$template_default}"
        else
            read -r -p "  value: " value
        fi
        set_env_value "$var" "$value"
        filled_vars+=("$var")
    fi
done 3< <(template_vars)

echo ""
if [ "$missing_count" -eq 0 ]; then
    print_success ".env has all variables from .env.template set"
elif $QUIET; then
    print_error "$missing_count variable(s) missing or empty in .env (run without --quiet to fill interactively)"
    exit 1
else
    print_success "Filled in ${#filled_vars[@]} variable(s): ${filled_vars[*]}"
fi
