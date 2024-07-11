#include <iostream>
#include <fstream>
#include <sstream>

using namespace TFEM;
using namespace std;

void TFEM::compute_stiff_matrix(HostMesh host_mesh, DeviceMesh device_mesh, Kokkos::View<double*> stiff, Kokkos::View<double*> coeffs, double k)
{
  int num_elems = host_mesh.n_regions;
  int num_points = host_mesh.n_points;
  // initally, set all entries of stiff vector to zero
  Kokkos::parallel_for(num_points, KOKKOS_LAMBDA (const int i) {
    stiff(i) = 0;
  });
  Kokkos::fence();
  // iterate over all triangles
  Kokkos::parallel_for(num_elems, KOKKOS_LAMBDA (const int i) {
    Kokkos::View<double[3][3]> pts("pts");
    for (int j = 0; j < 3; j++) {
      // get index of the j-th point in the triangle
      pts(j,0) = device_mesh.regions(i)[j];
      // get x-coordinate of the relevant point
      pts(j,1) = device_mesh.points(pts(j,0))[0];
      // get y-coordinate of the relevant points
      pts(j,2) = device_mesh.points(pts(j,0))[1];
    }
    // compute |J| for this triangle
    double jacob = 0.5*((pts(3,1)-pts(2,1))*(pts(1,2)-pts(2,2))-(pts(1,1)-pts(2,1))*(pts(3,2)-pts(2,2)));
    // compute entries in the vector S*c^n, where S is the stiffness matrix and c^n is the vector of coefficients from the n-th time step
    double dx_de = 0.5*(pts(0,1)-pts(1,1));
    double dx_dn = 0.5*(pts(2,1)-pts(1,1));
    double dy_de = 0.5*(pts(0,2)-pts(1,2));
    double dy_dn = 0.5*(pts(2,2)-pts(1,2));
    double du_de = 0.5*(coeffs(pts(2,0))-coeffs(pts(1,0)));
    double du_dn = 0.5*(coeffs((pts(0,0)))-coeffs(pts(1,0)));
    double du_dx = (0.25/jacob)*(dy_dn*du_de-dy_de*du_dn);
    double du_dy = (0.25/jacob)*(-dx_dn*du_de+dx_de*du_dn);
    for (int j = 0; j < 3: j++) {
      double dp_dx;
      double dp_dy;
      if (j == 0) {
        dp_dx = (0.5/jacob)*(-dy_de);
        dp_dy = (0.5/jacob)*(dx_de);
      } else if (j == 1) {
        dp_dx = (0.5/jacob)*(-dy_dn+dy_de);
        dp_dy = (0.5/jacob)*(dx_dn-dx_de);
      }
      else {
        dp_dx = (0.5/jacob)*(dy_dn);
        dp_dy = (0.5/jacob)*(-dx_dn);
      }
      double c = 2*jacob*(dp_dx*du_dx+dp_dy*du_dy);
      Kokkos::atomic_add(stiff(pts(j,0)), c);
    }
  });
  Kokkos::fence();
}
