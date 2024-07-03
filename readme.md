# High Order Triangular Finite Element Method- Approximate Mass Matrix
This project was created to demonstrate a parallel implementation of a new method for high-order finite element analysis on an unstructured triangular mesh. It is written in Kokkos, to provide the greatest portability- it can be run on CPU or on GPU.

## Building and Installing
This project requires the Kokkos library and the Kokkos Kernels library. It was written and tested with Kokkos 4.3.01 and KokkosKernels 4.3.01, but other releases likely work just as well. For a quick installation, look at `install_kokkos.sh`, which will download, build, and install Kokkos to `~/kokkos_build/`. It builds separate installs for CPU and GPU, though there may be ways of getting around this.

Once Kokkos is installed, the project is build and executed as normal using CMake- the main `CMakeLists.txt` is located under `highOrderTFEM` along with the rest of the source code. For convenience, scripts `build_gpu.sh` and `build_cpu.sh` have been provided. Once built, the demo executable will be located at `bin/gpu/demo` or `bin/cpu/demo`. 