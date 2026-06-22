#!/usr/bin/env bash
set -euo pipefail

TOML_FILE="pixi.toml"

TRUE_MARK="# local:true"
FALSE_MARK="# local:false"

PIXI_DIR=".pixi"
LOCK_FILE="pixi.lock"

fail() {
    echo "Error: $1" >&2
    exit 1
}

update_toml() {
    local state="$1" # true | false

    awk -v true_mark="$TRUE_MARK" -v false_mark="$FALSE_MARK" -v state="$state" '
        function is_commented(line) {
            return line ~ /^[[:space:]]*#/
        }

        function toggle_on(line) {
            sub(/^[[:space:]]*#[[:space:]]?/, "", line)
            return line
        }

        function toggle_off(line) {
            return is_commented(line) ? line : "# " line
        }

        {
            has_true_mark  = ($0 ~ true_mark)
            has_false_mark = ($0 ~ false_mark)
            needs_toggle = has_true_mark || has_false_mark

            if (needs_toggle) {
                toggle_switch = (state == "true" && has_true_mark) || (state == "false" && has_false_mark)
                if (toggle_switch) {
                    print toggle_on($0)
                } else {
                    print toggle_off($0)
                }
                next
            }

            print
        }
    ' "$TOML_FILE" > "$TOML_FILE.tmp"

    # Only replace if changed
    if cmp -s "$TOML_FILE" "$TOML_FILE.tmp"; then
        rm "$TOML_FILE.tmp"
        return 1  # no change
    else
        mv "$TOML_FILE.tmp" "$TOML_FILE"
        return 0  # changed
    fi
}

stash_state() {
    local label="$1"  # "true" or "false"

    if [[ -d "$PIXI_DIR" || -f "$LOCK_FILE" ]]; then
        echo "Stashing current state → $label"

        [[ -d "$PIXI_DIR" ]] && mv "$PIXI_DIR" ".pixi_local_${label}" 2>/dev/null || true
        [[ -f "$LOCK_FILE" ]] && mv "$LOCK_FILE" ".pixi_local_${label}.lock" 2>/dev/null || true
    fi
}

restore_state() {
    local label="$1"  # "true" or "false"

    if [[ -d ".pixi_local_${label}" || -f ".pixi_local_${label}.lock" ]]; then
        echo "Restoring cached state ← $label"

        [[ -d ".pixi_local_${label}" ]] && mv ".pixi_local_${label}" "$PIXI_DIR"
        [[ -f ".pixi_local_${label}.lock" ]] && mv ".pixi_local_${label}.lock" "$LOCK_FILE"
        return 0
    else
        echo "No cached state for $label (will require solve)"
        return 1
    fi
}

seed_true_from_false() {
    # Only allowed warm start direction: false → true
    if [[ -d ".pixi_local_false" && ! -d "$PIXI_DIR" ]]; then
        echo "Seeding local:true from local:false (hardlink rsync warm start)"

        mkdir -p "$PIXI_DIR"

        rsync -a --delete --link-dest="$(pwd)/.pixi_local_false" ".pixi_local_false/" "$PIXI_DIR/"
    fi
    if [[ -f ".pixi_local_false.lock" && ! -f "$LOCK_FILE" ]]; then
        cp ".pixi_local_false.lock" "$LOCK_FILE"
    fi
}

case "${1:-}" in
    start)
        # switch to local:true
        if update_toml "true"; then
            stash_state "false"

            # try restore, otherwise warm start from false
            restore_state "true" || seed_true_from_false
        else
            echo "Already in local:true state; nothing to do"
        fi
        ;;
    stop)
        # switch to local:false (canonical, no seeding needed)
        if update_toml "false"; then
            stash_state "true"
            restore_state "false"
        else
            echo "Already in local:false state; nothing to do"
        fi
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac
