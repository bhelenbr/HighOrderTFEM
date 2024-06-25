#!/bin/bash
cd ~
mkdir kokkos_build
cd kokkos_build

mkdir gpu_build
mkdir cpu_build
mkdir kokkos_gpu_install
mkdir kokkos_cpu_install

wget https://github.com/kokkos/kokkos/archive/refs/tags/4.3.01.tar.gz
tar -xvf 4.3.01.tar.gz

# CPU
cmake -S $(realpath ./kokkos-4.3.01/) -B $(realpath ./kokkos_cpu_install) -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_INSTALL_PREFIX=$(realpath ./cpu_build) -DKokkos_ENABLE_CUDA=OFF -DKokkos_ARCH_VOLTA70=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_SKX=ON -DKokkos_ENABLE_TEST=ON

# GPU
cmake -S $(realpath ./kokkos-4.3.01/) -B $(realpath ./kokkos_gpu_install) -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_INSTALL_PREFIX=$(realpath ./cpu_build) -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_SKX=ON -DCMAKE_CXX_COMPILER=$(realpath ./kokkos-4.3.01/bin/nvcc_wrapper) -DKokkos_ENABLE_TEST=ON

cd kokkos_cpu_install
make
make install
cd ..

cd kokkos_gpu_install
make
make install
cd ..

