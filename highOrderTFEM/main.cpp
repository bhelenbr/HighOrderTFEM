#include <iostream>

#include <Kokkos_Core.hpp>
#include <mesh.hpp>
#include <fem.hpp>
#include <iomanip>

int main(int argc, char *argv[])
{
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

        TFEM::MeshColorMap coloring(device_mesh);
        std::cout << "Colored into " << coloring.color_count() << " partitions" << std::endl;
        std::cout << "Sizes:";
        for (int color = 0; color < coloring.color_count(); color++)
        {
            std::cout << " " << coloring.member_count(color);
            assert(coloring.color_member_ids_host(color).extent(0) == coloring.member_count(color));
        }
        std::cout << std::endl;
        TFEM::validate_mesh_coloring(host_mesh, coloring);

        std::cout << "Mesh size: " << host_mesh.point_count()
                  << " " << host_mesh.edge_count()
                  << " " << host_mesh.region_count()
                  << std::endl;

        std::cout << "Mesh boundary segments:" << std::endl;
        for (int seg = 0; seg < host_mesh.boundary_edges.numRows(); seg++)
        {
            std::cout << "\t Seg " << (seg + 1) << ": ";
            auto segment = host_mesh.boundary_edges.rowConst(seg);
            for (int i = 0; i < segment.length; i++)
            {
                std::cout << " " << segment(i);
            }
            std::cout << std::endl;
        }

        TFEM::SolutionWriter writer("out/slices.json", host_mesh);
        TFEM::Solver solver(device_mesh, 1E-2, 1E-2);

        auto point_weight_mirror = Kokkos::create_mirror(solver.current_point_weights);
        Kokkos::deep_copy(point_weight_mirror, solver.current_point_weights);
        writer.add_slice(point_weight_mirror);

        auto timer = Kokkos::Timer();
        double start_time = timer.seconds();

        for (int i = 0; i < 10; i++)
        {
            solver.simulate_steps(1000);
            Kokkos::deep_copy(point_weight_mirror, solver.current_point_weights);
            Kokkos::fence();

            writer.add_slice(point_weight_mirror);
        }

        double stop_time = timer.seconds();

        Kokkos::deep_copy(point_weight_mirror, solver.current_point_weights);

        std::cout << "Final point 0 weight: " << std::setprecision(15) << point_weight_mirror(0) << std::endl;
        std::cout << "10000 step time (s): " << (stop_time - start_time) << std::endl;
    }
    Kokkos::finalize();
}