#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_dir="$(cd "$script_dir/../.." && pwd)"
mt4g_bin="${MT4G_BIN:-$repository_dir/build/mt4g}"
sweeps="${SWEEPS:-12}"

[[ "$sweeps" =~ ^[1-9][0-9]*$ ]] || {
    echo "SWEEPS must be a positive integer: $sweeps" >&2
    exit 2
}

module purge
module load "${ROCM_MODULE:-rocm/7.2}"

for required_command in rocminfo amd-smi python3 sha256sum; do
    command -v "$required_command" >/dev/null || {
        echo "Required command unavailable: $required_command" >&2
        exit 1
    }
done
[[ -x "$mt4g_bin" ]] || { echo "MT4G executable unavailable: $mt4g_bin" >&2; exit 1; }

# Keep the complete ROCr device enumeration. Device selection happens through
# MT4G's --device-id and hipSetDevice.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export HSA_XNACK=1

job_id="${SLURM_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)}"
node_name="$(hostname -s)"
job_dir="$script_dir/runs/cpx_$job_id"
node_dir="$job_dir/nodes/$node_name"

if [[ -e "$node_dir" ]]; then
    echo "Refusing to overwrite existing node output: $node_dir" >&2
    exit 1
fi
mkdir -p "$node_dir"

git_commit="$(git -C "$repository_dir" rev-parse HEAD 2>/dev/null || echo unknown)"

order_seed="${ORDER_SEED:-auto}"
if [[ "$order_seed" == "auto" ]]; then
    order_seed="$job_id"
fi

{
    echo "mode=cpx"
    echo "memory_partition=NPS1"
    echo "xnack=1"
    echo "allocator=hipmalloc"
    echo "hostname=$node_name"
    echo "job_id=$job_id"
    echo "step_id=${SLURM_STEP_ID:-unknown}"
    echo "sweeps=$sweeps"
    echo "devices=12"
    echo "packages=2"
    echo "order_design=williams"
    echo "order_seed=$order_seed"
    echo "timestamp_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "mt4g_bin=$mt4g_bin"
    echo "mt4g_sha256=$(sha256sum "$mt4g_bin" | awk '{print $1}')"
    echo "git_commit=$git_commit"
    echo "rocm_module=${ROCM_MODULE:-rocm/7.2}"
} >"$node_dir/metadata.txt"

capture_provenance() {
    local target="$1"
    mkdir -p "$target"

    rocminfo >"$target/rocminfo.txt" 2>&1
    amd-smi static --partition >"$target/amd_smi_partition.txt" 2>&1
    amd-smi static --asic >"$target/amd_smi_asic.txt" 2>&1 || true
    amd-smi version >"$target/amd_smi_version.txt" 2>&1 || true
    amd-smi firmware >"$target/amd_smi_firmware.txt" 2>&1 || true
    amd-smi topology >"$target/amd_smi_topology.txt" 2>&1 || true
    uname -a >"$target/uname.txt"
    (numactl --hardware || true) >"$target/numactl_hardware.txt" 2>&1
    (lscpu || true) >"$target/lscpu.txt" 2>&1
    (cat /proc/self/status || true) >"$target/process_status.txt" 2>&1
    (find /sys/class/kfd/kfd/topology/nodes -maxdepth 3 -type f \
        -name properties -print -exec cat {} \; || true) \
        >"$target/kfd_topology.txt" 2>&1
    env | LC_ALL=C sort >"$target/environment.txt"
}

create_device_map() {
    local provenance_dir="$1"
    local output="$2"
    local logical_device=0

    printf 'logical_device\tpackage\tpackage_device\tkfd_node\tlocation_id\tunique_id\trender_minor\tsimd_count\tnum_xcc\n' \
        >"$output"

    while IFS= read -r kfd_node_dir; do
        local properties="$kfd_node_dir/properties"
        [[ -f "$properties" ]] || continue

        local simd_count
        simd_count="$(awk '$1=="simd_count" {print $2}' "$properties")"
        [[ "${simd_count:-0}" =~ ^[0-9]+$ ]] || continue
        ((simd_count > 0)) || continue

        local location_id unique_id render_minor num_xcc package package_device
        location_id="$(awk '$1=="location_id" {print $2}' "$properties")"
        unique_id="$(awk '$1=="unique_id" {print $2}' "$properties")"
        render_minor="$(awk '$1=="drm_render_minor" {print $2}' "$properties")"
        num_xcc="$(awk '$1=="num_xcc" {print $2}' "$properties")"
        package=$((logical_device / 6))
        package_device=$((logical_device % 6))

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$logical_device" "$package" "$package_device" \
            "$(basename "$kfd_node_dir")" "$location_id" "$unique_id" \
            "$render_minor" "$simd_count" "$num_xcc" >>"$output"
        logical_device=$((logical_device + 1))
    done < <(find /sys/class/kfd/kfd/topology/nodes -mindepth 1 -maxdepth 1 \
        -type d -print | LC_ALL=C sort -V)

    [[ "$logical_device" -eq 12 ]] || {
        echo "KFD mapping mismatch: required=12 GPU nodes observed=$logical_device" >&2
        return 1
    }

    local kfd_locations rocm_locations
    kfd_locations="$(awk -F '\t' 'NR > 1 {print $5}' "$output" | paste -sd, -)"
    rocm_locations="$(awk '
        /Device Type:[[:space:]]*GPU/{gpu=1; next}
        gpu && /BDFID:/{print $NF; gpu=0}
        ' "$provenance_dir/rocminfo.txt" | paste -sd, -)"
    [[ "$kfd_locations" == "$rocm_locations" ]] || {
        echo "Logical-device mapping mismatch: KFD=$kfd_locations ROCr=$rocm_locations" >&2
        return 1
    }
}

validate_snapshot() {
    local provenance_dir="$1"
    local device_map="$2"

    local observed_mode memory_mode gpu_count cu_values xcc_values
    observed_mode="$(awk -F: '
        /(ACCELERATOR|COMPUTE)_PARTITION:/ {
            gsub(/[[:space:]]/, "", $2)
            value=tolower($2)
            if (value ~ /^(spx|tpx|cpx)$/) print value
        }' "$provenance_dir/amd_smi_partition.txt" | LC_ALL=C sort -u | paste -sd, -)"
    memory_mode="$(awk -F: '
        /MEMORY_PARTITION:/ {
            gsub(/[[:space:]]/, "", $2)
            value=toupper($2)
            if (value ~ /^NPS[1248]$/) print value
        }' "$provenance_dir/amd_smi_partition.txt" | LC_ALL=C sort -u | paste -sd, -)"
    gpu_count="$(awk '/Device Type:[[:space:]]*GPU/{count++} END{print count+0}' \
        "$provenance_dir/rocminfo.txt")"
    cu_values="$(awk '
        /Device Type:[[:space:]]*GPU/{gpu=1; next}
        gpu && /Compute Unit:/{print $NF; gpu=0}
        ' "$provenance_dir/rocminfo.txt" | LC_ALL=C sort -nu | paste -sd, -)"
    xcc_values="$(awk -F '\t' 'NR > 1 {print $9}' "$device_map" | LC_ALL=C sort -nu | paste -sd, -)"

    [[ "$observed_mode" == "cpx" ]] || {
        echo "Compute partition mismatch: required=cpx observed=${observed_mode:-unknown}" >&2
        return 1
    }
    [[ "$memory_mode" == "NPS1" ]] || {
        echo "Memory partition mismatch: required=NPS1 observed=${memory_mode:-unknown}" >&2
        return 1
    }
    [[ "$gpu_count" -eq 12 ]] || {
        echo "GPU-agent mismatch: required=12 observed=$gpu_count" >&2
        return 1
    }
    [[ "$cu_values" == "38" ]] || {
        echo "CU mismatch: required=38 per CPX agent observed=${cu_values:-unknown}" >&2
        return 1
    }
    [[ "$xcc_values" == "1" ]] || {
        echo "XCC mismatch: required=1 per CPX agent observed=${xcc_values:-unknown}" >&2
        return 1
    }
}

before_dir="$node_dir/provenance_before"
capture_provenance "$before_dir"
create_device_map "$before_dir" "$node_dir/devices.tsv"
validate_snapshot "$before_dir" "$node_dir/devices.tsv"

python3 "$script_dir/generate_orders.py" \
    --output "$node_dir/orders.tsv" \
    --repetitions "$sweeps" \
    --devices 12 \
    --seed "$order_seed" \
    --node "$node_name"

# Materialize every order before starting any benchmark so an interrupted job
# still retains the predeclared design.
for ((repeat = 1; repeat <= sweeps; ++repeat)); do
    repeat_dir="$node_dir/$(printf 'repeat_%02d' "$repeat")"
    mkdir -p "$repeat_dir"
    awk -F '\t' -v repeat="$repeat" 'NR == 1 || $1 == repeat' \
        "$node_dir/orders.tsv" >"$repeat_dir/order.txt"
done

printf 'repeat\tposition\tdevice\tpackage\tpackage_device\tstate\texit_status\tduration_seconds\tresult\n' \
    >"$node_dir/status.tsv"

failures=0

run_one() {
    local repeat="$1"
    local position="$2"
    local device="$3"
    local package="$4"
    local package_device="$5"
    local output_dir="$6"
    shift 6
    local -a benchmark_command=("$@")

    mkdir -p "$output_dir"
    printf '%q ' "${benchmark_command[@]}" >"$output_dir/command.txt"
    printf '\n' >>"$output_dir/command.txt"

    local started finished duration status state result_path
    started="$(date +%s)"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] node=$node_name repeat=$repeat position=$position device=$device package=$package"
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
        echo "repeat=$repeat"
        echo "position=$position"
        echo "device=$device"
        echo "package=$package"
        echo "package_device=$package_device"
        echo "state=$state"
        echo "exit_status=$status"
        echo "duration_seconds=$duration"
        echo "result=$result_path"
    } >"$output_dir/status.txt"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$repeat" "$position" "$device" "$package" "$package_device" \
        "$state" "$status" "$duration" "$result_path" \
        >>"$node_dir/status.tsv"
}

while IFS=$'\t' read -r repeat position device package package_device block design_seed; do
    repeat_dir="$node_dir/$(printf 'repeat_%02d' "$repeat")"
    device_dir="$repeat_dir/device_$(printf '%02d' "$device")"
    mkdir -p "$device_dir"
    awk -F '\t' -v device="$device" 'NR == 1 || $1 == device' \
        "$node_dir/devices.tsv" >"$device_dir/device.tsv"

    command=(
        "$mt4g_bin"
        --device-id "$device"
        --l1
        --l2
        --l3
        --shared
        --memory
        --optimal
        --raw
        --report
        --timing
        --quiet
        --location "$device_dir"
        --file result
    )
    run_one "$repeat" "$position" "$device" "$package" "$package_device" \
        "$device_dir" "${command[@]}"
done < <(awk -F '\t' 'NR > 1' "$node_dir/orders.tsv")

after_dir="$node_dir/provenance_after"
capture_provenance "$after_dir"
create_device_map "$after_dir" "$node_dir/devices_after.tsv"
if ! validate_snapshot "$after_dir" "$node_dir/devices_after.tsv"; then
    failures=$((failures + 1))
fi
if ! cmp -s "$node_dir/devices.tsv" "$node_dir/devices_after.tsv"; then
    diff -u "$node_dir/devices.tsv" "$node_dir/devices_after.tsv" \
        >"$node_dir/device_mapping_change.diff" || true
    echo "Device mapping changed during the campaign on $node_name" >&2
    failures=$((failures + 1))
fi

{
    echo "timestamp_end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "failures=$failures"
} >>"$node_dir/metadata.txt"

echo "RQ3 CPX sweep finished on $node_name: $node_dir"
if [[ "$failures" -ne 0 ]]; then
    echo "$failures validation or benchmark failure(s) occurred" >&2
    exit 1
fi
