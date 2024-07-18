#!/bin/bash
mkdir accuracy_tests
./build_gpu.sh

for i in $(seq 2 11);
do
    sbatch --wait --output=accuracy_tests/test_${i}.out ./run_gpu.slurm.sh demoMeshes/Results2/square${i}_b0.grd
done