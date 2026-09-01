#!/bin/env sh
set -e

# Start backend
python3 -m multi_camera.backend.fastapi &

cd /Mocap/frontend/acquisition

# REACT_APP_* values get baked in at build time. The image was built with
# defaults; rebuild only if the runtime env vars differ (e.g. mocap-test
# sets REACT_APP_TEST_MODE=true).
ENV_FINGERPRINT="${REACT_APP_BASE_URL:-localhost}|${REACT_APP_TEST_MODE:-}"
FINGERPRINT_FILE="build/.env-fingerprint"
if [ ! -f "$FINGERPRINT_FILE" ] || [ "$(cat "$FINGERPRINT_FILE")" != "$ENV_FINGERPRINT" ]; then
    echo "Frontend env-var fingerprint changed — rebuilding bundle…"
    npm run build
    echo "$ENV_FINGERPRINT" > "$FINGERPRINT_FILE"
fi

npm run serve