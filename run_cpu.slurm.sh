#!/bin/bash
#SBATCH --partition=general
#SBATCH --time=01:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=0
#SBATCH --job-name=CPU_FEM
#SBATCH --output=logs/cpu_%j.out

date

export OMP_PROC_BIND=spread
export OMP_PLACES=threads
./bin/cpu/demo "$@"


echo "Done!"
date