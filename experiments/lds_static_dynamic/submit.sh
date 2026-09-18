#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="$script_dir/job.sh"
action="${1:---dry-run}"
repetitions="${REPETITIONS:-10}"
allow_dirty="${ALLOW_DIRTY:-0}"
walltime="${WALLTIME:-01:00:00}"

case "$action" in
    --dry-run|--submit) ;;
    *) echo "Usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
esac

[[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || {
    echo "REPETITIONS must be a positive integer: $repetitions" >&2
    exit 2
}
[[ "$allow_dirty" =~ ^[01]$ ]] || {
    echo "ALLOW_DIRTY must be 0 or 1" >&2
    exit 2
}
[[ -x "$runner" ]] || { echo "Job script is not executable: $runner" >&2; exit 1; }

if ((repetitions % 2 != 0)); then
    echo "Warning: use an even REPETITIONS value to balance static/dynamic order." >&2
fi

mkdir -p "$script_dir/logs"

for mode in spx tpx cpx; do
    command=(
        sbatch
        --job-name="mt4g_lds_${mode}"
        --mi300-partition="$mode"
        --time="$walltime"
        --export="ALL,MODE=$mode,REPETITIONS=$repetitions,ALLOW_DIRTY=$allow_dirty"
        "$runner"
    )

    if [[ "$action" == "--submit" ]]; then
        "${command[@]}"
    else
        printf '%q ' "${command[@]}"
        printf '\n'
    fi
done
