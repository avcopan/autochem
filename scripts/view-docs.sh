#!/usr/bin/env bash

set -euo pipefail

if [[ -z "${BROWSER:-}" ]]; then
    echo "Error: The BROWSER environment variable is not set."
    echo "To view the documentation site, this variable must point"
    echo "to a web browser executable. For example:"
    echo "  export BROWSER='/mnt/c/Program Files/Mozilla Firefox/firefox.exe'"
    exit 1
fi

exec "$BROWSER" docs/build/index.html
