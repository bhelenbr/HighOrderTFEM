#include <iostream>

#include <Kokkos_Core.hpp>
#include <mesh.hpp>

int main(int argc, char *argv[]){
    Kokkos::initialize(argc, argv);
    {
        std::cout << "Running with default execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
        std::cout << "\tDefault host execution space: " << Kokkos::DefaultHostExecutionSpace::name() << std::endl;

        // Verify that we can read a mesh properly. Assume the file path is located as 
        // the first arg (after program name)
        TFEM::HostMesh host_mesh;
        TFEM::DeviceMesh device_mesh;
        TFEM::load_meshes_from_grd_file(argv[1], device_mesh, host_mesh);

        std::cout << "Host mesh size: " << host_mesh.point_count() 
                  << " " << host_mesh.edge_count() 
                  << " " << host_mesh.region_count()
                  << std::endl;
        std::cout << "Device mesh size: " << device_mesh.point_count() 
                  << " " << device_mesh.edge_count() 
                  << " " << device_mesh.region_count()
                  << std::endl;

        TFEM::pointID last_point = host_mesh.point_count()-1;
        std::cout << "Mesh point " << last_point << " coords: "
                  << host_mesh.point_coords(last_point, 0) << " "
                  << host_mesh.point_coords(last_point, 1) << std::endl;
        int last_edge = host_mesh.edge_count() - 1;
        std::cout << "Edge " << last_edge << " point ids: " 
                  << host_mesh.edge_to_point_ids(last_edge, 0) << " "
                  << host_mesh.edge_to_point_ids(last_edge, 1) << std::endl;
        int last_region = host_mesh.region_count() - 1;
        std::cout << "Triangle " << last_region << " point ids: " 
            << host_mesh.region_to_point_ids(last_region, 0) << " "
            << host_mesh.region_to_point_ids(last_region, 1) << " "
            << host_mesh.region_to_point_ids(last_region, 2) << std::endl;
    }
    Kokkos::finalize();
}