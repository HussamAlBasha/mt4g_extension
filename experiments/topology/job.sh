#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --constraint=apu
#SBATCH --gres=gpu:2
#SBATCH --time=00:05:00
#SBATCH --mail-type=none
#SBATCH --job-name=topology
#SBATCH --chdir=/u/halba/mt4g_extension/experiments/topology
#SBATCH --output=/u/halba/mt4g_extension/experiments/topology/logs/topology_%j.out

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "usage: sbatch --mi300-partition=MODE job.sh MODE" >&2
    exit 2
fi

mode="$1"
case "$mode" in
    spx|tpx|cpx) ;;
    *) echo "invalid mode: $mode" >&2; exit 2 ;;
esac

script_dir="/u/halba/mt4g_extension/experiments/topology"
worker="$script_dir/worker.sh"
[[ -x "$worker" ]] || { echo "worker is not executable: $worker" >&2; exit 1; }

srun --ntasks=1 --cpu-bind=cores "$worker" "$mode"
