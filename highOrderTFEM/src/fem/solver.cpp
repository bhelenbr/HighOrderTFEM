#include <Kokkos_Core.hpp>
#include "mesh.hpp"
#include "fem.hpp"

using namespace TFEM;

Solver::Solver(DeviceMesh mesh, double timestep, double k):
    mesh(mesh), dt(timestep), n_total_steps(0), k(k)
{
    current_point_weights = Kokkos::View<double*>("Current Point Weights", mesh.point_count());
    prev_point_weights = Kokkos::View<double*>("Prev Point Weights", mesh.point_count());
    point_mass_inv = Kokkos::View<double*>("Inverse Point Masses", mesh.point_count());
}

void Solver::setup_mass_matrix(){
    // TODO make this legit based on mesh
    // Right now give dummy scalar of all ones
    Kokkos::deep_copy(point_mass_inv, 1.0);
}


// DUMMY
void Solver::setup_initial(){
    // make up some bogus initial conditions
    
    // Manifest individual variables needed for capture-by-copy
    auto mesh = this->mesh;
    auto current_point_weights = this->current_point_weights;

    Kokkos::parallel_for(mesh.point_count(), KOKKOS_LAMBDA(int i){
        Point p = mesh.points(i);
        double x = p[0];
        double y = p[1];
        double weight = (1 - x * x) * (1 - y * y) * cos(10 * (x + y));
        current_point_weights(i) = weight;
    });
}


void Solver::simulate_steps(int n_steps){
    for(int i = 0; i < n_steps; (i++, n_total_steps++)){
        // Do step. TODO: we have at present a list of 
        // edges, meaning that each point is getting contribution from
        // perhaps multiple sources- a race condition. 
        // Typical solutions are:
        //   - Coloring algorithm
        //   - Atomic operations
        //   - Others?
        // Atomics are the easiest to program, but may have
        // performance implications. For now, go with atomics,
        // then benchmark vs (incorrect) free-for-all summation
        // to see how bad the hit is- if less than ~2x, it
        // may be better than coloring already.
        // see https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/Atomic-Operations.html
        // and https://kokkos.org/kokkos-core-wiki/API/core/atomics/atomic_op.html

        // Step one: copy current to previous. Now we can update current,
        // and it already has the identity term included- we just need to 
        // handle the increments
        Kokkos::deep_copy(prev_point_weights, current_point_weights);

        // Manifest individual variables needed for capture-by-copy
        auto mesh = this->mesh;
        auto point_mass_inv = this->point_mass_inv;
        auto prev_point_weights = this->prev_point_weights;
        auto current_point_weights = this->current_point_weights;
        auto k = this->k;
        auto dt = this->dt;

        // Now iterate over edges to add the contributions to each
        // neighboring point
        Kokkos::parallel_for(mesh.edge_count(), KOKKOS_LAMBDA(int i){
            Edge e = mesh.edges(i);

            // TODO: matrix values! figure these out
            double A_12 = 1.0;
            double A_21 = 1.0;

            double contribution_to_1 = -k * dt * point_mass_inv(e[0]) * A_12 * prev_point_weights(e[1]);
            double contribution_to_2 = -k * dt * point_mass_inv(e[1]) * A_21 * prev_point_weights(e[0]);

            Kokkos::atomic_add(&current_point_weights(e[0]), contribution_to_1);
            Kokkos::atomic_add(&current_point_weights(e[1]), contribution_to_2);
        });
    }
}