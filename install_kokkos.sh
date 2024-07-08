#!/bin/bash
cd ~
mkdir kokkos_build
cd kokkos_build

mkdir gpu_build
mkdir cpu_build
mkdir gpu_kernels_build
mkdir cpu_kernels_build
mkdir kokkos_gpu_install
mkdir kokkos_cpu_install
mkdir kokkos_kernels_gpu_install
mkdir kokkos_kernels_cpu_install

# Get Kokkos core
wget https://github.com/kokkos/kokkos/archive/refs/tags/4.3.01.tar.gz
tar -xvf 4.3.01.tar.gz

# Get Kokkos Kernels library
wget https://github.com/kokkos/kokkos-kernels/releases/download/4.3.01/kokkos-kernels_4.3.01.tar.gz 
tar -xvf kokkos-kernels_4.3.01.tar.gz

# CPU
cmake -S $(realpath ./kokkos-4.3.01/) -B $(realpath ./cpu_build) -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_INSTALL_PREFIX=$(realpath ./kokkos_cpu_install) -DKokkos_ENABLE_CUDA=OFF -DKokkos_ARCH_VOLTA70=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_SKX=ON -DKokkos_ENABLE_TEST=ON
cd cpu_build
make
make install
cd ..

# CPU kernels
cmake -S $(realpath ./kokkos-kernels) -B $(realpath ./cpu_kernels_build) -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_INSTALL_PREFIX=$(realpath ./kokkos_kernels_cpu_install) -DKokkos_ROOT=$(realpath ./kokkos_cpu_install)
cd cpu_kernels_build/
make
make install
cd ..

# GPU
cmake -S $(realpath ./kokkos-4.3.01/) -B $(realpath ./gpu_build) -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_INSTALL_PREFIX=$(realpath ./kokkos_gpu_install) -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_SKX=ON -DCMAKE_CXX_COMPILER=$(realpath ./kokkos-4.3.01/bin/nvcc_wrapper) -DKokkos_ENABLE_TEST=ON
cd gpu_build
make
make install
cd ..

# GPU kernels
cmake -S $(realpath ./kokkos-kernels) -B $(realpath ./gpu_kernels_build) -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_INSTALL_PREFIX=$(realpath ./kokkos_kernels_gpu_install) -DKokkos_ROOT=$(realpath ./kokkos_gpu_install) -DCMAKE_CXX_COMPILER=$(realpath ./kokkos-4.3.01/bin/nvcc_wrapper) -DKokkosKernels_REQUIRE_DEVICES=CUDA
cd gpu_kernels_build/
make
make install
cd ..
