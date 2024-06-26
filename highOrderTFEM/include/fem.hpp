#ifndef highOrderTFEM_fem_hpp
#define highOrderTFEM_fem_hpp

#include <string>
#include <fstream>

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

    class SolutionWriter{
        protected:
            std::ofstream out_file;
            DeviceMesh::HostMirrorMesh mesh;
            int slice_count;

        public:
            SolutionWriter(std::string fname, DeviceMesh::HostMirrorMesh mesh){
                slice_count = 0;
                this->mesh = mesh;
                out_file = std::ofstream(fname);
                out_file << "{\"points\": [";
                for(pointID p = 0; p < mesh.point_count(); p++){
                    if(p > 0){
                        out_file << ", ";
                    }
                    out_file << "[" << mesh.points(p)[0] << ", " << mesh.points(p)[1] << "]";
                }
                out_file << "],\n\"slices\":[";
            }
            
            template <class ViewType>
            void add_slice(ViewType view){
                static_assert(Kokkos::is_view_v<ViewType>, "ViewType must be view");
                static_assert(ViewType::Rank == 1, "Points are arranged as a flat grid");
                static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename ViewType::memory_space>::accessible, "View must be accessible from host");
                assert(view.extent(0) == mesh.point_count());

                if(slice_count > 0){
                    out_file << ",";
                }
                slice_count++;

                out_file << std::endl << "[";
                for(pointID p = 0; p < mesh.point_count(); p++){
                    if(p > 0){
                        out_file << ", ";
                    }
                    out_file << view(p);
                }
                out_file << "]";
            }

            ~SolutionWriter(){
                // need to write out end cap to file
                out_file << "]}";
            }
    };
}

#endif