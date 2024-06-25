export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export KOKKOS_SRC_DIR=$(realpath ~/kokkos_build/kokkos-4.3.01/)
export KOKKOS_INSTALL_DIR=$(realpath ~/kokkos_build/kokkos_cpu_install/)
cmake -S ./highOrderTFEM -B ./bin/cpu -DKokkos_ROOT=$KOKKOS_INSTALL_DIR 
cd bin/cpu 
make