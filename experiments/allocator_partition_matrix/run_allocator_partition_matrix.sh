#!/bin/bash -l
#SBATCH -o /u/halba/mt4g_extension/experiments/allocator_partition_matrix/logs/job_%x_%j.out
#SBATCH -D /u/halba/mt4g_extension/experiments/allocator_partition_matrix
#SBATCH -J mt4g_allocator_matrix
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=apu
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --mail-type=none
#SBATCH --time=01:00:00

set -euo pipefail

experiment_dir="/u/halba/mt4g_extension/experiments/allocator_partition_matrix"
worker="$experiment_dir/worker.sh"

[[ -x "$worker" ]] || { echo "Worker is not executable: $worker" >&2; exit 1; }

srun --nodes=3 --ntasks=3 --ntasks-per-node=1 --cpu-bind=cores "$worker"
