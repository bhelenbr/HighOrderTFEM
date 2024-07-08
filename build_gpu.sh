export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export KOKKOS_SRC_DIR=$(realpath ~/kokkos_build/kokkos-4.3.01/)
export KOKKOS_INSTALL_DIR=$(realpath ~/kokkos_build/kokkos_gpu_install/)
export KOKKOS_KERNELS_INSTALL_DIR=$(realpath ~/kokkos_build/kokkos_kernels_gpu_install/)
cmake -S ./highOrderTFEM -B ./bin/gpu -DKokkos_ROOT=$KOKKOS_INSTALL_DIR -DCMAKE_CXX_COMPILER=${KOKKOS_SRC_DIR}/bin/nvcc_wrapper -DKokkosKernels_ROOT=$KOKKOS_KERNELS_INSTALL_DIR
cd bin/gpu 
make