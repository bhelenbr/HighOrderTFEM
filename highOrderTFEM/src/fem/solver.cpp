#include <Kokkos_Core.hpp>
#include "mesh.hpp"
#include "solver.hpp"
#include "analytical.hpp"
#include <iostream>
#include "scatter_pattern.hpp"

using namespace TFEM;

template <typename ScatterPattern>
Solver<ScatterPattern>::Solver(DeviceMesh mesh, ScatterPattern pattern, Analytical::ZeroBoundary<> boundary_conditions, double timestep, double k)
    : mesh(mesh),
      dt(timestep),
      n_total_steps(0),
      k(k),
      current_point_weights("Current Point Weights", mesh.point_count()),
      prev_point_weights("Prev Point Weights", mesh.point_count()),
      point_mass_inv("Inverse Point Masses", mesh.point_count()),
      scatter_pattern(pattern),
      boundary(boundary_conditions)
{
    // By assigning the non-const view to the const view, we
    // essentialy point the const view to the same memory- they
    // will update in parallel.
    prev_point_weights_readonly = prev_point_weights;
    point_mass_inv_readonly = point_mass_inv;
    setup_mass_matrix();
    setup_initial_conditions();
    Kokkos::fence();
}

template <typename ScatterPattern>
void Solver<ScatterPattern>::setup_mass_matrix()
{
    // Create local var to avoid capturing the "this" pointer
    auto point_mass_inv = this->point_mass_inv;

    // Dispatch the mass matrix assembly functor to the scatter pattern.
    SolverImpl::MassMatrixFunctor<ScatterPattern> mass_functor(point_mass_inv, mesh, k, dt);
    scatter_pattern.distribute_work(mass_functor);

    // pre-invert the diagonal now to avoid an operation each timestep.
    Kokkos::parallel_for(mesh.point_count(), KOKKOS_LAMBDA(const int &i) { //
        point_mass_inv(i) = 1 / point_mass_inv(i);
    });
    Kokkos::fence();
}

template <typename ScatterPattern>
void Solver<ScatterPattern>::setup_initial_conditions()
{
    // Manifest individual variables needed for capture-by-copy to avoid capturing
    // the "this" pointer.
    auto point_coords = this->mesh.points;
    auto current_points = this->current_point_weights;
    auto boundary = this->boundary;

    Kokkos::parallel_for(mesh.point_count(), KOKKOS_LAMBDA(int i) {
        Point p = point_coords(i);
        double x = p[0];
        double y = p[1];
        current_points(i) = boundary(x, y, 0); });
}

template <typename ScatterPattern>
void Solver<ScatterPattern>::simulate_steps(int n_steps)
{
    for (int i = 0; i < n_steps; i++)
    {
        n_total_steps++;
        prepare_next_step();
        Kokkos::fence();
        compute_step();
        Kokkos::fence();
        fix_boundary();
        Kokkos::fence();
    }
}

template <typename ScatterPattern>
void Solver<ScatterPattern>::prepare_next_step()
{
    // When we move to the next step, the current state becomes the previous state.
    Kokkos::deep_copy(prev_point_weights, current_point_weights);
}

template <typename ScatterPattern>
void Solver<ScatterPattern>::compute_step()
{
    // Dispatch element-wise contributions. The identity-matrix term already handled
    // as a precondition to calling this function.
    SolverImpl::ElementContributionFunctor<ScatterPattern> per_element_functor(current_point_weights, prev_point_weights_readonly, point_mass_inv_readonly, mesh, k, dt);
    scatter_pattern.distribute_work(per_element_functor);
}

template <typename ScatterPattern>
void Solver<ScatterPattern>::fix_boundary()
{
    auto mesh = this->mesh;
    auto current_points = this->current_point_weights;
    // Set all points on a boundary edge to 0.
    Kokkos::parallel_for(mesh.boundary_edge_count(), KOKKOS_LAMBDA(int i) {
        auto edge_id = mesh.boundary_edges.entries(i);
        auto e = mesh.edges(edge_id);
        
        // This will likely double-set every boundary point, but is more robust
        // against errors like one edge of a segment being reversed and a point
        // not getting set at all.
        current_points(e[0]) = 0; 
        current_points(e[1]) = 0; });
}

template <typename ScatterPattern>
double Solver<ScatterPattern>::measure_error()
{
    double t = time();
    auto mesh = this->mesh;
    auto current_points = this->current_point_weights;
    auto analytic = this->boundary;
    double interior_result = 0;
    Kokkos::parallel_reduce(mesh.point_count(), KOKKOS_LAMBDA(const int &i, double &err_sum) {
        if(!mesh.boundary_points(i)) { // only compute error for interior
            Point p = mesh.points(i);
            double numerical_value = current_points(i);
            double analytic_value = analytic(p[0], p[1], t);
            err_sum += pow(analytic_value - numerical_value, 2);} }, interior_result);

    // Boundary points are held to correct, so they contribute 0 error
    // and don't really give a good idea of whats going on in the interior.
    // Measure error on interior points only.
    return (interior_result) / (mesh.point_count() - mesh.n_boundary_points);
}

KOKKOS_INLINE_FUNCTION double det_jacobian(Point pts[3])
{
    return 0.25 * ((pts[2][0] - pts[1][0]) * (pts[0][1] - pts[1][1]) - (pts[0][0] - pts[1][0]) * (pts[2][1] - pts[1][1]));
}

template <typename ScatterPattern>
KOKKOS_INLINE_FUNCTION void SolverImpl::MassMatrixFunctor<ScatterPattern>::operator()(Region element) const
{
    // Fetch coordinates for the element
    Point pts[3];
    for (int j = 0; j < 3; j++)
    {
        pts[j] = mesh.points(element[j]);
    }

    // compute |J| for this triangle
    double jacob = det_jacobian(pts);

    // compute mass-lumped entries for the inverse of the mass matrix.
    // The contribution is the sum of the main |J|/3 diagonal plus two |J|/6 off-diagonals for this element,
    // and the contribution is the same for each point. (In the linear case).
    double c = (jacob * 2 / 3);
    for (int j = 0; j < 3; j++)
    {
        ScatterPattern::contribute(&inv_mass(element[j]), c);
    }
}

template <typename ScatterPattern>
KOKKOS_INLINE_FUNCTION void SolverImpl::ElementContributionFunctor<ScatterPattern>::operator()(Region element) const
{
    // Fetch coordinates for the element
    Point pts[3];
    for (int j = 0; j < 3; j++)
    {
        pts[j] = mesh.points(element[j]);
    }

    // compute |J| for this triangle
    double jacob = det_jacobian(pts);

    // compute entries in the vector S*c^n, where S is the stiffness matrix and c^n is the vector of coefficients from the n-th time step
    // Calculate change-of-variable partial derivatives
    double dx_de = 0.5 * (pts[2][0] - pts[1][0]); // point 2 switched with 0
    double dx_dn = 0.5 * (pts[0][0] - pts[1][0]); // ""
    double dy_de = 0.5 * (pts[2][1] - pts[1][1]); // ""
    double dy_dn = 0.5 * (pts[0][1] - pts[1][1]); // ""
    // Calculate gradients
    double du_de = 0.5 * (prev_points(element[2]) - prev_points(element[1]));
    double du_dn = 0.5 * (prev_points(element[0]) - prev_points(element[1]));
    double du_dx = (1 / jacob) * (dy_dn * du_de - dy_de * du_dn);
    double du_dy = (1 / jacob) * (-dx_dn * du_de + dx_de * du_dn);
    // Based on gradients, calculate interaction
    for (int j = 0; j < 3; j++)
    {
        double dp_dx;
        double dp_dy;
        switch (j)
        {
        case 0:
            dp_dx = (0.5 / jacob) * (-dy_de);
            dp_dy = (0.5 / jacob) * (dx_de);
            break;
        case 1:
            dp_dx = (0.5 / jacob) * (-dy_dn + dy_de);
            dp_dy = (0.5 / jacob) * (dx_dn - dx_de);
            break;
        case 2:
            dp_dx = (0.5 / jacob) * (dy_dn);
            dp_dy = (0.5 / jacob) * (-dx_dn);
        }

        double c = 2 * jacob * (dp_dx * du_dx + dp_dy * du_dy);
        double contribution = -k * dt * inv_mass(element[j]) * c;
        ScatterPattern::contribute(&new_points(element[j]), contribution);
    }
}

// We need to specify what classes we might be using so the linker doesn't get mad
template class Solver<ColoredElementScatterAdd>;
template class Solver<AtomicElementScatterAdd>;
template class Solver<SerialElementScatterAdd>;