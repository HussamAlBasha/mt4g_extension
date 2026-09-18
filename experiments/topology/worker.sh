#!/usr/bin/env bash
set -euo pipefail

mode="${1:?usage: worker.sh spx|tpx|cpx}"
case "$mode" in
    spx|tpx|cpx) ;;
    *) echo "invalid mode: $mode" >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module purge
module load "${ROCM_MODULE:-rocm/7.2}"

command -v rocminfo >/dev/null || { echo "rocminfo unavailable" >&2; exit 1; }
command -v amd-smi >/dev/null || { echo "amd-smi unavailable" >&2; exit 1; }

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$script_dir/runs/$mode/${stamp}_j${SLURM_JOB_ID:-manual}"
mkdir -p "$run_dir"

# Keep collecting if an optional command is unavailable or unsupported.
rocminfo >"$run_dir/rocminfo.txt" 2>&1
amd-smi static --partition >"$run_dir/amd_smi_partition.txt" 2>&1
amd-smi static --asic >"$run_dir/amd_smi_asic.txt" 2>&1 || true
amd-smi version >"$run_dir/amd_smi_version.txt" 2>&1 || true
amd-smi firmware >"$run_dir/amd_smi_firmware.txt" 2>&1 || true
amd-smi topology >"$run_dir/amd_smi_topology.txt" 2>&1 || true
uname -a >"$run_dir/uname.txt"
(numactl --hardware || true) >"$run_dir/numactl_hardware.txt" 2>&1
(lscpu || true) >"$run_dir/lscpu.txt" 2>&1
(cat /proc/self/status || true) >"$run_dir/process_status.txt" 2>&1
(find /sys/class/kfd/kfd/topology/nodes -maxdepth 3 -type f \
    -name properties -print -exec cat {} \; || true) \
    >"$run_dir/kfd_topology.txt" 2>&1
env | LC_ALL=C sort >"$run_dir/environment.txt"

observed_mode="$(awk -F: '
    /(ACCELERATOR|COMPUTE)_PARTITION:/ {
        gsub(/[[:space:]]/, "", $2)
        value=tolower($2)
        if (value ~ /^(spx|tpx|cpx)$/) print value
    }' "$run_dir/amd_smi_partition.txt" | LC_ALL=C sort -u | paste -sd, -)"

cat >"$run_dir/summary.txt" <<EOF
requested_mode=$mode
observed_mode=${observed_mode:-unknown}
hostname=$(hostname)
job_id=${SLURM_JOB_ID:-manual}
step_id=${SLURM_STEP_ID:-unknown}
timestamp_utc=$stamp
EOF

if [[ "$observed_mode" != "$mode" ]]; then
    echo "partition mismatch: requested=$mode observed=${observed_mode:-unknown}" >&2
    echo "snapshot retained in $run_dir" >&2
    exit 1
fi

echo "topology snapshot written to $run_dir"
