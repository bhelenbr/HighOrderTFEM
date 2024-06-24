#include <iostream>

#include <Kokkos_Core.hpp>
#include <mesh.hpp>

int main(int argc, char *argv[]){
    Kokkos::initialize(argc, argv);
    {
        std::cout << "Running with default execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
        std::cout << "\tDefault host execution space: " << Kokkos::DefaultHostExecutionSpace::name() << std::endl;
        dummy_mesh_fn();
    }
    Kokkos::finalize();
}