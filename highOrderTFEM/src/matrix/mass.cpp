#include <iostream>
#include <fstream>
#include <sstream>
#include "mesh.hpp"

using namespace TFEM;
using namespace std;

// host_mesh and device_mesh should be already initialized (and identical)
void TFEM::compute_mass_matrix(HostMesh host_mesh, DeviceMesh device_mesh, Kokkos::View<double*> mass, double delta_t)
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
    double jacob = 0.5*((pts(2,1)-pts(1,1))*(pts(0,2)-pts(1,2))-(pts(0,1)-pts(1,1))*(pts(2,2)-pts(1,2)));                                         
    // compute mass-lumped entries for the inverse of the mass matrix
    double c = delta_t / (jacob*(2/3));
    for (int j = 0; j < 3; j++) {
      Kokkos::atomic_add(mass(pts(j,0)), c);
    }
  });
  Kokkos::fence();
}
