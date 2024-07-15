#!/bin/bash

./build_gpu.sh

sbatch --wait ./run_gpu.slurm.sh demoMeshes/square1.grd
sbatch --wait ./run_gpu.slurm.sh demoMeshes/square2.grd
sbatch --wait ./run_gpu.slurm.sh demoMeshes/square3.grd
sbatch --wait ./run_gpu.slurm.sh demoMeshes/square4.grd
sbatch --wait ./run_gpu.slurm.sh demoMeshes/square5.grd
sbatch --wait ./run_gpu.slurm.sh demoMeshes/square5.grd
sbatch --wait ./run_gpu.slurm.sh demoMeshes/square6.grd