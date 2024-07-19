#include <iostream>
#include <iomanip>

#define COLOR 0
#define ATOMIC 1
#define SERIAL 2
// Use this define statement to select algorithm
#define SCATTER_ALGO COLOR

#include <Kokkos_Core.hpp>
#include <mesh.hpp>
#include <solver.hpp>
#include <analytical.hpp>
#include "scatter_pattern.hpp"

int main(int argc, char *argv[])
{
    assert(argc > 1); // Must provide mesh input file
    Kokkos::initialize(argc, argv);
    {
        std::cout << "Running with default execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;
        std::cout << "\tDefault host execution space: " << Kokkos::DefaultHostExecutionSpace::name() << std::endl;
        std::cout << "Concurrency: " << Kokkos::DefaultExecutionSpace::concurrency() << std::endl;

        // Verify that we can read a mesh properly. Assume the file path is located as
        // the first arg (after program name)
        TFEM::DeviceMesh device_mesh;
        TFEM::DeviceMesh::HostMirrorMesh host_mesh;
        bool fuzz = false;
        TFEM::load_meshes_from_grd_file(argv[1], device_mesh, host_mesh, fuzz);

        // Analytic solution and output writer
        // Create an analytical solution to test against
        double k = 1E-2;
        double dt = 1E-5;
        std::vector<TFEM::Analytical::Term> terms;
        terms.push_back({1.0, 1, 1});
        terms.push_back({2.0, 1, 3});
        TFEM::Analytical::ZeroBoundary<> analytical(k, -1.0, 2.0, -1.0, 2.0, terms);

// Depending on macros, either create a coloring-based or atomic-based solver.
#if SCATTER_ALGO == COLOR
        // Find element coloring.
        TFEM::MeshColorMap coloring(device_mesh);

        // Print coloring debug info, since it's nondeterministic and affects runtime
        std::cout << "Colored into " << coloring.color_count() << " partitions" << std::endl;
        std::cout << "Sizes:";
        for (int color = 0; color < coloring.color_count(); color++)
        {
            std::cout << " " << coloring.member_count(color);
            assert(coloring.color_member_ids_host(color).extent(0) == coloring.member_count(color));
        }
        std::cout << std::endl;

        TFEM::ColoredElementScatterAdd scatter_pattern(coloring);

        TFEM::Solver<TFEM::ColoredElementScatterAdd> solver(device_mesh, scatter_pattern, analytical, dt, k);
#elif SCATTER_ALGO == ATOMIC
        TFEM::AtomicElementScatterAdd scatter_pattern(device_mesh);

        TFEM::Solver<TFEM::AtomicElementScatterAdd> solver(device_mesh, scatter_pattern, analytical, dt, k);
#elif SCATTER_ALGO == SERIAL
        TFEM::SerialElementScatterAdd scatter_pattern(device_mesh);

        TFEM::Solver<TFEM::SerialElementScatterAdd> solver(device_mesh, scatter_pattern, analytical, dt, k);
#endif // end of use_color if-else

        // Initialize writer
        TFEM::SolutionWriter writer("out/slices.json", host_mesh);
        auto point_weight_mirror = Kokkos::create_mirror(solver.current_point_weights);
        Kokkos::deep_copy(point_weight_mirror, solver.current_point_weights);
        writer.add_slice(point_weight_mirror);
        Kokkos::fence();

        std::cout << "Starting simulation" << std::endl;

        auto timer = Kokkos::Timer();
        double start_time = timer.seconds();

        for (int i = 0; i < 10; i++)
        {
            solver.simulate_steps(1000);
            Kokkos::deep_copy(point_weight_mirror, solver.current_point_weights);
            Kokkos::fence();
            std::cout << "Root mean square error: " << sqrt(solver.measure_error()) << std::endl;

            writer.add_slice(point_weight_mirror);
            Kokkos::fence();
        }

        double stop_time = timer.seconds();

        Kokkos::deep_copy(point_weight_mirror, solver.current_point_weights);

        std::cout << "10000 step time (s): " << (stop_time - start_time) << std::endl;
    }
    Kokkos::finalize();
}
