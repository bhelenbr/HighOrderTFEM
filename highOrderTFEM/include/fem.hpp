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
            Kokkos::View<double*> point_mass_inv;

            /**
             * TODO: right now this is a dummy
            */
            void setup_mass_matrix();

        public:
            Kokkos::View<double*> current_point_weights;
            Kokkos::View<double*> prev_point_weights;

            Solver(DeviceMesh mesh, double timestep, double k);

            // This is a dummy until we can figure out initial conditions
            void setup_initial();

            void simulate_steps(int n_steps);
    };
}

#endif