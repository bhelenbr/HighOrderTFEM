#!/bin/bash

./build_cpu.sh

sbatch ./run_cpu.slurm.sh demoMeshes/square1.grd
sbatch ./run_cpu.slurm.sh demoMeshes/square2.grd
sbatch ./run_cpu.slurm.sh demoMeshes/square3.grd
sbatch ./run_cpu.slurm.sh demoMeshes/square4.grd
sbatch ./run_cpu.slurm.sh demoMeshes/square5.grd
sbatch ./run_cpu.slurm.sh demoMeshes/square6.grd