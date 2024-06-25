#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=GPU_FEM
#SBATCH --output=logs/gpu_%j.out
date 
# Print the GPU indices assigned to me
echo "Given gpu device indices: $SLURM_JOB_GPUS" 

# Do my actual job here...
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
./bin/gpu/demo "$@"

echo "Done!"

date