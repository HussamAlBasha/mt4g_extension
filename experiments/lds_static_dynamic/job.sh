#!/bin/bash -l
#SBATCH --output=/u/halba/mt4g_extension/experiments/lds_static_dynamic/logs/job_%x_%j.out
#SBATCH --chdir=/u/halba/mt4g_extension/experiments/lds_static_dynamic
#SBATCH --job-name=mt4g_lds
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=apu
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --mail-type=none
#SBATCH --time=01:00:00

set -euo pipefail

experiment_dir="/u/halba/mt4g_extension/experiments/lds_static_dynamic"
worker="$experiment_dir/worker.sh"

[[ -x "$worker" ]] || { echo "Worker is not executable: $worker" >&2; exit 1; }

srun --nodes=3 --ntasks=3 --ntasks-per-node=1 \
    --cpu-bind=cores --kill-on-bad-exit=0 \
    --output="$experiment_dir/logs/worker_${SLURM_JOB_ID}_%N.out" \
    --error="$experiment_dir/logs/worker_${SLURM_JOB_ID}_%N.err" \
    "$worker"
