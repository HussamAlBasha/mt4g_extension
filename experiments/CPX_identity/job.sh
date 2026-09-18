#!/bin/bash -l
#SBATCH --job-name=cpx_identity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=apu
#SBATCH --gres=gpu:2
#SBATCH --exclusive
#SBATCH --time=00:05:00
#SBATCH --mail-type=none
#SBATCH --output=identity_%j.out
set -euo pipefail

# Submit from this experiment directory, with --nodelist set explicitly.
cd "$SLURM_SUBMIT_DIR"
module purge
module load "${ROCM_MODULE:-rocm/7.2}"
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export HSA_XNACK=1
srun --ntasks=1 --cpu-bind=cores python3 identity.py capture
