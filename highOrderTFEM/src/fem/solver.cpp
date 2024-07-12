#include <Kokkos_Core.hpp>
#include "mesh.hpp"
#include "fem.hpp"
#include "analytical.hpp"
#include <iostream>

using namespace TFEM;

Solver::Solver(DeviceMesh mesh, MeshColorMap color, Analytical::ZeroBoundary<> boundary_conditions, double timestep, double k)
    : mesh(mesh),
      dt(timestep),
      n_total_steps(0),
      k(k),
      current_point_weights("Current Point Weights", mesh.point_count()),
      prev_point_weights("Prev Point Weights", mesh.point_count()),
      point_mass_inv("Inverse Point Masses", mesh.point_count()),
      element_coloring(color),
      boundary(boundary_conditions)
{
    // point the readonly buffers to the correct place.
    // mass matrix readonly set up alongisde mass matrix itself
    prev_point_weights_readonly = prev_point_weights;
    setup_mass_matrix();
    setup_initial_conditions();
    Kokkos::fence();
}

KOKKOS_INLINE_FUNCTION double det_jacobian(Point pts[3])
{
    return 0.25 * ((pts[2][0] - pts[1][0]) * (pts[0][1] - pts[1][1]) - (pts[0][0] - pts[1][0]) * (pts[2][1] - pts[1][1]));
}

void Solver::setup_mass_matrix()
{
    // TODO make this legit based on mesh
    // Right now give dummy scalar of all ones

    auto mesh = this->mesh;
    auto point_mass_inv = this->point_mass_inv;
    this->point_mass_inv_readonly = point_mass_inv;

    int num_elems = mesh.region_count();
    int num_points = mesh.point_count();

    for (int color = 0; color < element_coloring.color_count(); color++)
    {
        auto color_elements = element_coloring.color_member_regions(color);
        Kokkos::parallel_for(color_elements.extent(0), KOKKOS_LAMBDA(const int &i) {
            Region element = color_elements(i);
            Point pts[3];
            for (int j = 0; j < 3; j++)
            {
                pts[j] = mesh.points(element[j]);
            }
            // compute |J| for this triangle
            double jacob = det_jacobian(pts);
            // compute mass-lumped entries for the inverse of the mass matrix
            double c = (jacob * 2 / 3);
            for (int j = 0; j < 3; j++)
            {
                point_mass_inv(element[j]) += c;
            } });
        Kokkos::fence();
    }

    Kokkos::parallel_for(num_points, KOKKOS_LAMBDA(const int &i) { //
        point_mass_inv(i) = 1 / point_mass_inv(i);
    });
    Kokkos::fence();
}

// DUMMY
void Solver::setup_initial_conditions()
{
    // make up some bogus initial conditions

    // Manifest individual variables needed for capture-by-copy
    auto mesh = this->mesh;
    auto current_points = this->current_point_weights;
    auto boundary = this->boundary;

    Kokkos::parallel_for(mesh.point_count(), KOKKOS_LAMBDA(int i) {
        Point p = mesh.points(i);
        double x = p[0];
        double y = p[1];
        current_points(i) = boundary(x, y, 0); });
}

void Solver::simulate_steps(int n_steps)
{
    for (int i = 0; i < n_steps; (i++, n_total_steps++))
    {
        Kokkos::fence();
        prepare_next_step();
        Kokkos::fence();
        compute_step();
        Kokkos::fence();
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
        auto elements = element_coloring.color_member_regions(color);

        Kokkos::parallel_for(elements.extent(0), KOKKOS_LAMBDA(int i) {
            Region element = elements(i);
            do_element(element); });
        Kokkos::fence();
    }
}

void Solver::fix_boundary()
{
    auto mesh = this->mesh;
    auto current_points = this->current_point_weights;
    // For now, disregard how edges are segmented and set to 0.
    Kokkos::parallel_for(mesh.boundary_edge_count(), KOKKOS_LAMBDA(int i) {
        auto edge_id = mesh.boundary_edges.entries(i);
        auto e = mesh.edges(edge_id);

        current_points(e[0]) = 0; // Assume that the boundary is closed and uniformly directed, so this hits every boundary point
    });
}

double Solver::measure_error()
{
    double t = n_total_steps * dt;
    auto mesh = this->mesh;
    auto current_points = this->current_point_weights;
    auto analytic = this->boundary;
    double all_result = 0;
    Kokkos::parallel_reduce(mesh.point_count(), KOKKOS_LAMBDA(const int &i, double &err_sum) {
        Point p = mesh.points(i);
        double numerical_value = current_points(i);
        double analytic_value = analytic(p[0], p[1], t);
        err_sum += pow(analytic_value - numerical_value, 2); }, all_result);

    // Boundary points are held to correct, so they contribute 0 error
    // but don't really give a good idea of whats going on in the interior.
    // Measure only interior points.
    return (all_result) / (mesh.point_count() - mesh.boundary_edge_count());
}

SolverImpl::ElementContributionFunctor Solver::create_element_contribution_functor()
{
    return SolverImpl::ElementContributionFunctor(current_point_weights, prev_point_weights_readonly, point_mass_inv_readonly, mesh, k, dt);
}

KOKKOS_INLINE_FUNCTION void SolverImpl::ElementContributionFunctor::operator()(Region element) const
{
    Point pts[3];
    for (int j = 0; j < 3; j++)
    {
        pts[j] = mesh.points(element[j]);
    }
    // compute |J| for this triangle
    double jacob = det_jacobian(pts);
    // compute entries in the vector S*c^n, where S is the stiffness matrix and c^n is the vector of coefficients from the n-th time step
    double dx_de = 0.5 * (pts[0][0] - pts[1][0]);
    double dx_dn = 0.5 * (pts[2][0] - pts[1][0]);
    double dy_de = 0.5 * (pts[0][1] - pts[1][1]);
    double dy_dn = 0.5 * (pts[2][1] - pts[1][1]);
    double du_de = 0.5 * (prev_points(element[2]) - prev_points(element[1]));
    double du_dn = 0.5 * (prev_points(element[0]) - prev_points(element[1]));
    double du_dx = (1 / jacob) * (dy_dn * du_de - dy_de * du_dn);
    double du_dy = (1 / jacob) * (-dx_dn * du_de + dx_de * du_dn);
    for (int j = 0; j < 3; j++)
    {
        double dp_dx;
        double dp_dy;
        if (j == 0)
        {
            dp_dx = (0.5 / jacob) * (-dy_de);
            dp_dy = (0.5 / jacob) * (dx_de);
        }
        else if (j == 1)
        {
            dp_dx = (0.5 / jacob) * (-dy_dn + dy_de);
            dp_dy = (0.5 / jacob) * (dx_dn - dx_de);
        }
        else
        {
            dp_dx = (0.5 / jacob) * (dy_dn);
            dp_dy = (0.5 / jacob) * (-dx_dn);
        }
        double c = 2 * jacob * (dp_dx * du_dx + dp_dy * du_dy);
        if (element[j] == 15102) {
          printf("dp/dx: %f,   du/dx: %f,   dp_dy: %f,   du_dy: %f\n",dp_dx,du_dx,dp_dy,du_dy):
          printf("c: %f, |J|: %f, gradients: %f\n", c, jacob, dp_dx * du_dx + dp_dy * du_dy);
        }
        new_points(element[j]) += -k * dt * inv_mass(element[j]) * c;
    }
}
