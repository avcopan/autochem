#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [on|off]"
  exit 1
}

# Ensure exactly one argument
[[ $# -eq 1 ]] || usage

case "$1" in
  off)
    git config core.hooksPath /dev/null
    echo "Git hooks disabled for this repository."
    ;;
  on)
    git config --unset core.hooksPath
    echo "Git hooks re-enabled for this repository."
    ;;
  *)
    usage
    ;;
esac
