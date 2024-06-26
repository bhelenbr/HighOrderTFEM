#ifndef highOrderTFEM_fem_hpp
#define highOrderTFEM_fem_hpp

#include <Kokkos_Core.hpp>
#include "mesh.hpp"

namespace TFEM {

    class Solver{
        protected:
            DeviceMesh mesh;
            double dt;
            double k;
            int n_total_steps;

            // For the diagonal psuedomass matrix, this stores the 
            // inverse of the diagonal entries- saving a division 
            // each time.
            Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> point_mass_inv;


        public:
            Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> current_point_weights;
            Kokkos::View<double*, Kokkos::MemoryTraits<Kokkos::RandomAccess>> prev_point_weights;

            Solver(DeviceMesh mesh, double timestep, double k);

            void simulate_steps(int n_steps);

            // I am for some strange lambda-compilation-related reason forced to
            // make these public, but they are already called inside the constructor:
            // no need to call explicitly.
             /**
             * TODO: right now this is a dummy
            */
            void setup_mass_matrix();
            // This is a dummy until we can figure out initial conditions
            void setup_initial();
    };
}

#endif