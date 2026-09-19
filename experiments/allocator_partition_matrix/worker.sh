#!/usr/bin/env bash

set -uo pipefail

mode="${MODE:?MODE must be spx, tpx, or cpx}"
xnack="${XNACK:?XNACK must be 0 or 1}"
experiment_dir="/u/halba/mt4g_extension/experiments/allocator_partition_matrix"
repository_dir="/u/halba/mt4g_extension"
mt4g_bin="${MT4G_BIN:-$repository_dir/build/mt4g}"
repetitions="${REPETITIONS:-5}"

case "$mode" in
    spx|tpx|cpx) ;;
    *) echo "Invalid MODE: $mode" >&2; exit 2 ;;
esac

case "$xnack" in
    0|1) ;;
    *) echo "Invalid XNACK: $xnack" >&2; exit 2 ;;
esac

if [[ ! "$repetitions" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid REPETITIONS: $repetitions" >&2
    exit 2
fi

module purge
module load "${ROCM_MODULE:-rocm/7.2}"
[[ -x "$mt4g_bin" ]] || { echo "Missing executable: $mt4g_bin" >&2; exit 1; }

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export HSA_XNACK="$xnack"

job_id="${SLURM_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)}"
job_dir="$experiment_dir/runs/${mode}_xnack${xnack}_${job_id}"
node_name="$(hostname -s)"
node_dir="$job_dir/nodes/$node_name"
mkdir -p "$node_dir"

{
    echo "mode=$mode"
    echo "xnack=$xnack"
    echo "hsa_xnack=$HSA_XNACK"
    echo "hostname=$node_name"
    echo "job_id=$job_id"
    echo "step_id=${SLURM_STEP_ID:-unknown}"
    echo "repetitions=$repetitions"
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "mt4g_bin=$mt4g_bin"
    echo "git_commit=$(git -C "$repository_dir" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "rocm_module=${ROCM_MODULE:-rocm/7.2}"
} >"$node_dir/metadata.txt"

env | LC_ALL=C sort >"$node_dir/environment.txt"

partition_file="$node_dir/amd_smi_partition.txt"
if ! amd-smi static --partition >"$partition_file" 2>&1; then
    echo "Could not query the active compute partition with amd-smi" >&2
    exit 1
fi

observed_mode="$(awk -F: '
    /(ACCELERATOR|COMPUTE)_PARTITION:/ {
        gsub(/[[:space:]]/, "", $2)
        value=tolower($2)
        if (value ~ /^(spx|tpx|cpx)$/) print value
    }' "$partition_file" | LC_ALL=C sort -u | paste -sd, -)"

if [[ "$observed_mode" != "$mode" ]]; then
    echo "Compute partition mismatch: requested=$mode observed=${observed_mode:-unknown}" >&2
    exit 1
fi
echo "observed_mode=$observed_mode" >>"$node_dir/metadata.txt"

printf 'repeat\tallocator\tstate\texit_status\tresult\n' >"$node_dir/status.tsv"
failures=0

# Reproduce the coverage of the retained final campaign. SPX and TPX use
# MT4G's default AMD selection. CPX omits only the scalar group, which did not
# complete reliably in that mode, while retaining every other AMD group.
benchmark_groups=()
if [[ "$mode" == "cpx" ]]; then
    benchmark_groups=(
        --l1
        --l2
        --l3
        --shared
        --memory
        --departuredelay
        --resourceshare
    )
fi

for ((repeat = 1; repeat <= repetitions; ++repeat)); do
    repeat_name="$(printf 'repeat_%02d' "$repeat")"

    for allocator in hipmalloc hipmallocmanaged hiphostmalloc malloc; do
        run_dir="$node_dir/$repeat_name/$allocator"
        mkdir -p "$run_dir"

        command=(
            "$mt4g_bin"
            --device-id 0
            "${benchmark_groups[@]}"
            --optimal
            --static
            --allocator "$allocator"
            --graphs
            --raw
            --report
            --timing
            --quiet
            --location "$run_dir"
            --file result
        )

        printf '%q ' "${command[@]}" >"$run_dir/command.txt"
        printf '\n' >>"$run_dir/command.txt"

        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] mode=$mode xnack=$xnack node=$node_name repeat=$repeat allocator=$allocator"
        status=0
        "${command[@]}" >"$run_dir/stdout.log" 2>"$run_dir/stderr.log" || status=$?

        result_path="$run_dir/result.json"
        if [[ "$status" -eq 0 && -f "$result_path" ]]; then
            state="ok"
        elif [[ "$status" -eq 0 ]]; then
            state="missing-result"
            status=3
            failures=$((failures + 1))
        elif [[ "$allocator" == "malloc" && "$xnack" == "0" ]]; then
            state="expected-failure"
        else
            state="failed"
            failures=$((failures + 1))
        fi

        {
            echo "allocator=$allocator"
            echo "mode=$mode"
            echo "xnack=$xnack"
            echo "node=$node_name"
            echo "repeat=$repeat"
            echo "state=$state"
            echo "exit_status=$status"
            echo "result=$result_path"
        } >"$run_dir/status.txt"

        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$repeat" "$allocator" "$state" "$status" "$result_path" \
            >>"$node_dir/status.tsv"
    done
done

echo "Results for $node_name: $node_dir"
if [[ "$failures" -ne 0 ]]; then
    echo "$failures unexpected allocator run(s) failed" >&2
    exit 1
fi
