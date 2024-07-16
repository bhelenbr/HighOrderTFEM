# High Order Triangular Finite Element Method- Approximate Mass Matrix
This project was created to demonstrate a parallel implementation of a new method for high-order finite element analysis on an unstructured triangular mesh. It is written in Kokkos, to provide the greatest portability- it can be run on CPU or on GPU.

## Building and Installing
This project requires the Kokkos library and the Kokkos Kernels library. It was written and tested with Kokkos 4.3.01 and KokkosKernels 4.3.01, but other releases likely work just as well. For a quick installation, look at `install_kokkos.sh`, which will download, build, and install Kokkos to `~/kokkos_build/`. It builds separate installs for CPU and GPU, though there may be ways of getting around this.

Once Kokkos is installed, the project is build and executed as normal using CMake- the main `CMakeLists.txt` is located under `highOrderTFEM` along with the rest of the source code. For convenience, scripts `build_gpu.sh` and `build_cpu.sh` have been provided. Once built, the demo executable will be located at `bin/gpu/demo` or `bin/cpu/demo`. 

## Software Design Overview

This project consists of a solver class and various supporting classes, the usage of which are demonstrated in `demo.cpp`. The important components are:
 * Mesh and mesh coloring, provided in `mesh.hpp`. Includes reading a (triangular!) mesh from an input file and access to the mesh. The mesh coloring finds a (non-minimal) partitioning/coloring of mesh triangles such that triangles that share a point have different colors, for use in handling concurrency issues.
 * Closed-term solutions and test cases, provided in `analytical.hpp`
 * A somewhat generic means of handling a scatter pattern in `scatter_pattern.hpp`, where a work function is distributed onto compute units but functions have an overlapping write set. Currently supports atomic and coloring-based scatter-add patterns.
 * And finally, `solver.hpp`, which implements an explicit mass-lumped 1st order finite element method for the heat equation. 

 ### A Guide to Scatter Patterns

 Most solutions to distributed write conflicts involve either some tweaking with how work is distributed (such as coloring) or modifications to the write operations (such as atomic operations or mutexes). As such, our abstraction of a scatter patterrn constists of a function that can take in an arbitrary functor and distribute it accross the computing domain, and a function for performing a specific contribution operation. 

 The distribution function is templated for an arbitrary functor so that the pattern can be reused with different work loads. For the pattern to work, each functor must respect its `contribute()` operation. To make implementation easier and avoid cyclic template dependencies while still permitting pattern-generic functors, the contribute operation is made static. Thus, a functor can template on the pattern class to gain access to it's implementation of `contribute()`, and the functor type itself is used to specialize the `distribute_work()` function.

 ### The Solver Implementation

 The process the solver takes is roughly as follows:
  1. Initialize the mass matrix: use the provided scatter pattern to compute the contributes each element makes to the lumped mass matrix, then when that is done invert the diagonal.
  2. Initialize the initial conditions: match the provided analytical solution for $t = 0$. No need to use a scatter pattern as each thread writes only to one point, with no overlap.
  3. For each time step:
        1. Copy the current state to the previous state
        2. Use the scatter pattern to add contributions to the current state in-place, treating all points as interior points. (By not wiping the current state, the $Iu^n$ term in $u^{n+1} = (I+M^{-1}A)u^n$ is implicitly taken care of.)
        3. Go back and fix the boundary points, which were treated as interior points at the previous step