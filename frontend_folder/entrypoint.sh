#!/bin/sh
set -e

# Generate /usr/share/nginx/html/js/config.js at runtime if API_BASE_URL provided
CONFIG_PATH="/usr/share/nginx/html/js/config.js"

cat > "$CONFIG_PATH" <<EOF
// Auto-generated at container start
window.FRAUD_API_BASE_URL = "${API_BASE_URL}";
EOF

echo "[entrypoint] Wrote runtime config with API_BASE_URL=${API_BASE_URL}" >&2

exec "$@"
