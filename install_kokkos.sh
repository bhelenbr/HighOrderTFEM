#!/bin/bash
cd ~
mkdir kokkos_build
cd kokkos_build

mkdir kokkos_install

wget https://github.com/kokkos/kokkos/archive/refs/tags/4.3.01.tar.gz
tar -xvf 4.3.01.tar.gz

cmake -S $(realpath ./kokkos-4.3.01/) -B $(realpath ./kokkos_install) -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_INSTALL_PREFIX=$(pwd) -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ARCH_SKX=ON -DCMAKE_CXX_COMPILER=$(pwd)/kokkos-4.3.01/bin/nvcc_wrapper -DKokkos_ENABLE_TEST=ON -DKokkos_ENABLE_BENCHMARKS=ON -DKokkos_ENABLE_EXAMPLES=ON

cd kokkos_install
make
make install

