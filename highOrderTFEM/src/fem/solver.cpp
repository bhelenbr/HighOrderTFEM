#include <Kokkos_Core.hpp>
#include "mesh.hpp"
#include "fem.hpp"

using namespace TFEM;

Solver::Solver(DeviceMesh mesh, double timestep, double k)
    : mesh(mesh),
      dt(timestep),
      n_total_steps(0),
      k(k),
      current_point_weights("Current Point Weights", mesh.point_count()),
      prev_point_weights("Prev Point Weights", mesh.point_count()),
      point_mass_inv("Inverse Point Masses", mesh.point_count()),
      element_coloring(mesh)
{
    // point the readonly buffers to the correct place.
    // mass matrix readonly set up alongisde mass matrix itself
    prev_point_weights_readonly = prev_point_weights;
    setup_mass_matrix();
    setup_initial_conditions();
}

void Solver::setup_mass_matrix()
{
    // TODO make this legit based on mesh
    // Right now give dummy scalar of all ones
    Kokkos::deep_copy(point_mass_inv, 1.0);
    point_mass_inv_readonly = point_mass_inv;
}

// DUMMY
void Solver::setup_initial_conditions()
{
    // make up some bogus initial conditions

    // Manifest individual variables needed for capture-by-copy
    auto mesh = this->mesh;
    auto current_points = this->current_point_weights;

    Kokkos::parallel_for(mesh.point_count(), KOKKOS_LAMBDA(int i) {
        Point p = mesh.points(i);
        double x = p[0];
        double y = p[1];
        double weight = (1 - x * x) * (1 - y * y) * cos(10 * (x + y));
        current_points(i) = weight; });
}

void Solver::simulate_steps(int n_steps)
{
    for (int i = 0; i < n_steps; (i++, n_total_steps++))
    {
        prepare_next_step();
        compute_step();
        fix_boundary();
    }
}

void Solver::prepare_next_step()
{
    // Step one: copy current to previous. Now we can update current,
    // and it already has the identity term included- we just need to
    // handle the increments
    Kokkos::deep_copy(prev_point_weights, current_point_weights);
}

void Solver::compute_step()
{
    auto do_element = create_element_contribution_functor();
    for (int color = 0; color < element_coloring.color_count(); color++)
    {
        auto elements = element_coloring.color_view(color);

        Kokkos::parallel_for(elements.extent(0), KOKKOS_LAMBDA(int i) {
            Region element = elements(i);
            do_element(element); });
        Kokkos::fence();
    }
}

void Solver::fix_boundary()
{
    // TODO
}

SolverImpl::ElementContributionFunctor Solver::create_element_contribution_functor()
{
    return SolverImpl::ElementContributionFunctor(current_point_weights, prev_point_weights_readonly, point_mass_inv_readonly, mesh, k, dt);
}

KOKKOS_INLINE_FUNCTION void SolverImpl::ElementContributionFunctor::operator()(Region element) const
{
    // TODO put the magic here!
    // This is where the contribution for an element is calculated.
    // Right now I have made up a dummy.
    for (int edge_i = 0; edge_i < 3; edge_i++)
    {
        pointID p1 = element[edge_i];
        pointID p2 = element[(edge_i + 1 % 3)];

        double A12 = 1.0;
        double A21 = 1.0;

        new_points(p1) += kdt * inv_mass(p1) * A12 * prev_points(p2);
        new_points(p2) += kdt * inv_mass(p2) * A21 * prev_points(1);
        // .. or something like that anyway. Not too concerned at the moment.
    }
}