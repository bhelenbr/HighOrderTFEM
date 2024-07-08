#include <iostream>

#include <Kokkos_Core.hpp>
#include <mesh.hpp>
#include <fem.hpp>

int main(int argc, char *argv[]){
    assert(argc > 1); // Must provide mesh input file
    Kokkos::initialize(argc, argv);
    {  
        std::cout << "Running with default execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
        std::cout << "\tDefault host execution space: " << Kokkos::DefaultHostExecutionSpace::name() << std::endl;

        // Verify that we can read a mesh properly. Assume the file path is located as 
        // the first arg (after program name)
        TFEM::DeviceMesh device_mesh;
        TFEM::DeviceMesh::HostMirrorMesh host_mesh;
        TFEM::load_meshes_from_grd_file(argv[1], device_mesh, host_mesh);

        std::cout << "Mesh size: " << host_mesh.point_count() 
                  << " " << host_mesh.edge_count() 
                  << " " << host_mesh.region_count()
                  << std::endl;

        TFEM::SolutionWriter writer("out/slices.json", host_mesh);
        TFEM::Solver solver(device_mesh, 1E-2, 1E-2);

        auto point_weight_mirror = Kokkos::create_mirror_view(solver.current_point_weights);
        Kokkos::deep_copy(point_weight_mirror, solver.current_point_weights);
        writer.add_slice(point_weight_mirror);

        auto timer = Kokkos::Timer();
        double start_time = timer.seconds();

        for(int i = 0; i < 10; i++){
            solver.simulate_steps(1000);
            Kokkos::deep_copy(point_weight_mirror, solver.current_point_weights);
            writer.add_slice(point_weight_mirror);
        }

        double stop_time = timer.seconds();
        
        Kokkos::deep_copy(point_weight_mirror, solver.current_point_weights);
        
        std::cout << "Final point 0 weight: " << point_weight_mirror(0) << std::endl;
        std::cout << "10000 step time (s): " << (stop_time - start_time);
    }
    Kokkos::finalize();
}