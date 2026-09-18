#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd "$script_dir/../.." && pwd)"
mt4g_bin="${MT4G_BIN:-$repository_dir/build/mt4g}"
mode="${MODE:?MODE must be spx, tpx, or cpx}"
repetitions="${REPETITIONS:-10}"
allow_dirty="${ALLOW_DIRTY:-0}"

case "$mode" in
    spx|tpx|cpx) ;;
    *) echo "Invalid MODE: $mode" >&2; exit 2 ;;
esac
[[ "$repetitions" =~ ^[1-9][0-9]*$ ]] || {
    echo "REPETITIONS must be a positive integer: $repetitions" >&2
    exit 2
}
[[ "$allow_dirty" =~ ^[01]$ ]] || {
    echo "ALLOW_DIRTY must be 0 or 1" >&2
    exit 2
}

module purge
module load "${ROCM_MODULE:-rocm/7.2}"

for required_command in rocminfo amd-smi sha256sum; do
    command -v "$required_command" >/dev/null || {
        echo "Required command unavailable: $required_command" >&2
        exit 1
    }
done
[[ -x "$mt4g_bin" ]] || { echo "MT4G executable unavailable: $mt4g_bin" >&2; exit 1; }

# Keep ROCr's complete device enumeration. MT4G selects logical device 0 with
# hipSetDevice. XNACK is fixed because LDS does not use pageable system memory.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export HSA_XNACK=1

job_id="${SLURM_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)}"
node_name="$(hostname -s)"
job_dir="$script_dir/runs/${mode}_${job_id}"
node_dir="$job_dir/nodes/$node_name"

if [[ -e "$node_dir" ]]; then
    echo "Refusing to overwrite existing node output: $node_dir" >&2
    exit 1
fi
mkdir -p "$node_dir"

git -C "$repository_dir" status --short --untracked-files=all \
    >"$node_dir/git_status.txt" 2>&1 || true
git -C "$repository_dir" diff HEAD >"$node_dir/git_diff.patch" 2>&1 || true
git_commit="$(git -C "$repository_dir" rev-parse HEAD 2>/dev/null || echo unknown)"
git_clean=1
if [[ -s "$node_dir/git_status.txt" ]]; then
    git_clean=0
fi

rocminfo >"$node_dir/rocminfo.txt" 2>&1
amd-smi static --partition >"$node_dir/amd_smi_partition_before.txt" 2>&1
env | LC_ALL=C sort >"$node_dir/environment.txt"

observed_mode="$(awk -F: '
    /(ACCELERATOR|COMPUTE)_PARTITION:/ {
        gsub(/[[:space:]]/, "", $2)
        value=tolower($2)
        if (value ~ /^(spx|tpx|cpx)$/) print value
    }' "$node_dir/amd_smi_partition_before.txt" | LC_ALL=C sort -u | paste -sd, -)"
memory_mode="$(awk -F: '
    /MEMORY_PARTITION:/ {
        gsub(/[[:space:]]/, "", $2)
        value=toupper($2)
        if (value ~ /^NPS[1248]$/) print value
    }' "$node_dir/amd_smi_partition_before.txt" | LC_ALL=C sort -u | paste -sd, -)"
gpu_count="$(awk '/Device Type:[[:space:]]*GPU/{count++} END{print count+0}' \
    "$node_dir/rocminfo.txt")"
cu_values="$(awk '
    /Device Type:[[:space:]]*GPU/{gpu=1; next}
    gpu && /Compute Unit:/{print $NF; gpu=0}
    ' "$node_dir/rocminfo.txt" | LC_ALL=C sort -nu | paste -sd, -)"

case "$mode" in
    spx) expected_gpus=2; expected_cus=228 ;;
    tpx) expected_gpus=6; expected_cus=76 ;;
    cpx) expected_gpus=12; expected_cus=38 ;;
esac

[[ "$observed_mode" == "$mode" ]] || {
    echo "Compute partition mismatch: requested=$mode observed=${observed_mode:-unknown}" >&2
    exit 1
}
[[ "$memory_mode" == "NPS1" ]] || {
    echo "Memory partition mismatch: required=NPS1 observed=${memory_mode:-unknown}" >&2
    exit 1
}
[[ "$gpu_count" -eq "$expected_gpus" ]] || {
    echo "GPU-agent mismatch: required=$expected_gpus observed=$gpu_count" >&2
    exit 1
}
[[ "$cu_values" == "$expected_cus" ]] || {
    echo "CU mismatch: required=$expected_cus per agent observed=${cu_values:-unknown}" >&2
    exit 1
}

{
    echo "mode=$mode"
    echo "memory_partition=$memory_mode"
    echo "xnack=1"
    echo "device=0"
    echo "hostname=$node_name"
    echo "job_id=$job_id"
    echo "step_id=${SLURM_STEP_ID:-unknown}"
    echo "task_id=${SLURM_PROCID:-0}"
    echo "repetitions=$repetitions"
    echo "order_design=alternating-paired"
    echo "timestamp_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "mt4g_bin=$mt4g_bin"
    echo "mt4g_sha256=$(sha256sum "$mt4g_bin" | awk '{print $1}')"
    echo "git_commit=$git_commit"
    echo "git_clean=$git_clean"
    echo "rocm_module=${ROCM_MODULE:-rocm/7.2}"
    echo "observed_gpu_count=$gpu_count"
    echo "observed_cus_per_device=$cu_values"
} >"$node_dir/metadata.txt"

if [[ "$git_clean" -ne 1 && "$allow_dirty" -ne 1 ]]; then
    echo "Source tree is dirty; snapshot retained in $node_dir" >&2
    echo "Commit the benchmark source or set ALLOW_DIRTY=1 only for a smoke test." >&2
    exit 1
fi

printf 'repeat\tposition\tvariant\n' >"$node_dir/order.tsv"
printf 'repeat\tposition\tvariant\tstate\texit_status\tduration_seconds\tresult\n' \
    >"$node_dir/status.tsv"
failures=0
task_phase="${SLURM_PROCID:-0}"

run_one() {
    local repeat="$1"
    local position="$2"
    local variant="$3"
    local output_dir="$4"
    local -a benchmark_command=("${@:5}")

    mkdir -p "$output_dir"
    printf '%q ' "${benchmark_command[@]}" >"$output_dir/command.txt"
    printf '\n' >>"$output_dir/command.txt"

    local started finished duration status state result_path
    started="$(date +%s)"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] mode=$mode node=$node_name repeat=$repeat position=$position variant=$variant"
    status=0
    "${benchmark_command[@]}" >"$output_dir/stdout.log" 2>"$output_dir/stderr.log" || status=$?
    finished="$(date +%s)"
    duration=$((finished - started))
    result_path="$output_dir/result.json"

    if [[ "$status" -eq 0 && -f "$result_path" ]]; then
        state="ok"
    elif [[ "$status" -eq 0 ]]; then
        state="missing-result"
        status=3
        failures=$((failures + 1))
    else
        state="failed"
        failures=$((failures + 1))
    fi

    {
        echo "mode=$mode"
        echo "node=$node_name"
        echo "repeat=$repeat"
        echo "position=$position"
        echo "variant=$variant"
        echo "state=$state"
        echo "exit_status=$status"
        echo "duration_seconds=$duration"
        echo "result=$result_path"
    } >"$output_dir/status.txt"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$repeat" "$position" "$variant" "$state" "$status" "$duration" "$result_path" \
        >>"$node_dir/status.tsv"
}

for ((repeat = 1; repeat <= repetitions; ++repeat)); do
    # Every node alternates AB/BA. SLURM_PROCID offsets the first pair so all
    # nodes do not execute the same variant simultaneously.
    if (((repeat + task_phase) % 2 == 1)); then
        variants=(dynamic static)
    else
        variants=(static dynamic)
    fi

    for position in 1 2; do
        variant="${variants[position - 1]}"
        printf '%s\t%s\t%s\n' "$repeat" "$position" "$variant" >>"$node_dir/order.tsv"
        output_dir="$node_dir/$(printf 'repeat_%02d' "$repeat")/$variant"

        command=(
            "$mt4g_bin"
            --device-id 0
            --shared
            --optimal
        )
        if [[ "$variant" == "static" ]]; then
            command+=(--static)
        fi
        command+=(
            --raw
            --report
            --timing
            --quiet
            --location "$output_dir"
            --file result
        )

        run_one "$repeat" "$position" "$variant" "$output_dir" "${command[@]}"
    done
done

amd-smi static --partition >"$node_dir/amd_smi_partition_after.txt" 2>&1
observed_mode_after="$(awk -F: '
    /(ACCELERATOR|COMPUTE)_PARTITION:/ {
        gsub(/[[:space:]]/, "", $2)
        value=tolower($2)
        if (value ~ /^(spx|tpx|cpx)$/) print value
    }' "$node_dir/amd_smi_partition_after.txt" | LC_ALL=C sort -u | paste -sd, -)"
if [[ "$observed_mode_after" != "$mode" ]]; then
    echo "Compute partition changed during execution: expected=$mode observed=${observed_mode_after:-unknown}" >&2
    failures=$((failures + 1))
fi

{
    echo "timestamp_end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "failures=$failures"
} >>"$node_dir/metadata.txt"

echo "Results for $node_name: $node_dir"
if [[ "$failures" -ne 0 ]]; then
    echo "$failures LDS run(s) failed" >&2
    exit 1
fi
