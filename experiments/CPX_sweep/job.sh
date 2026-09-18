#!/bin/bash -l
#SBATCH --output=/u/halba/mt4g_extension/experiments/CPX_sweep/logs/job_%x_%j.out
#SBATCH --chdir=/u/halba/mt4g_extension/experiments/CPX_sweep
#SBATCH --job-name=mt4g_cpx_sweep
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=apu
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --mail-type=none
#SBATCH --time=08:00:00

set -euo pipefail

experiment_dir="/u/halba/mt4g_extension/experiments/CPX_sweep"
worker="$experiment_dir/worker.sh"

[[ -x "$worker" ]] || { echo "Worker is not executable: $worker" >&2; exit 1; }

srun --nodes=3 --ntasks=3 --ntasks-per-node=1 \
    --cpu-bind=cores --kill-on-bad-exit=0 \
    --output="$experiment_dir/logs/worker_${SLURM_JOB_ID}_%N.out" \
    --error="$experiment_dir/logs/worker_${SLURM_JOB_ID}_%N.err" \
    "$worker"
