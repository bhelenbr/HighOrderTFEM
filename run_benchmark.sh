#!/bin/bash
mkdir benchmark
./build_gpu.sh
./build_cpu.sh

for i in $(seq 2 11);
do
    sbatch --wait --mem=4GB --output=benchmark/cpu_${i}.out run_cpu.slurm.sh demoMeshes/Results2/square${i}_b0.grd
    sbatch --wait --output=benchmark/gpu_${i}.out run_gpu.slurm.sh demoMeshes/Results2/square${i}_b0.grd
    sbatch --wait --mem=4GB --cpus-per-task=1 --output=benchmark/serial_${i}.out run_cpu.slurm.sh demoMeshes/Results2/square${i}_b0.grd
done
