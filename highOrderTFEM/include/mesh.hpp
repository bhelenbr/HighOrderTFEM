/**
 * Triangular mesh datastructure, including file IO and copying to/from device.
*/
#ifndef highOrderTFEM_mesh_hpp
#define highOrderTFEM_mesh_hpp

#include <Kokkos_Core.hpp>
#include <string>


namespace TFEM {

    typedef uint16_t pointID;
    typedef double coordinate;

    template <class ExecSpace = Kokkos::DefaultExecutionSpace>
    class Mesh {
        // Create type aliases for clarity and for easy size tweaks (i.e. upgrade to long double)
        typedef Mesh<Kokkos::DefaultHostExecutionSpace> HostMesh;
        

        protected:
            pointID n_points;
            int n_edges;
            int n_regions;

        public:
            Kokkos::View<coordinate*[2], ExecSpace> point_coords;
            Kokkos::View<pointID*[2], ExecSpace> edge_to_point_ids;
            Kokkos::View<pointID*[3], ExecSpace> region_to_point_ids;
            
            /**
             * Sometimes need to leave an uninitialized mesh.
            */
            Mesh() = default;

            /**
             * Creates a mesh of given dimensions.
            */
            Mesh(int n_points, int n_edges, int n_regions) : 
                n_points(n_points), n_edges(n_edges), n_regions(n_regions),
                point_coords("mesh_point_coords", n_points),
                edge_to_point_ids("mesh_edge_points", n_edges),
                region_to_point_ids("mesh_region_points", n_regions)
            {
                // initializer list took care of most of it.
            }


            /**
             * Creates a copy of the mesh with all views accessible from the host.
             * 
             * Does NOT perform a deep copy- do manually or call src_mesh.deep_copy_all_to(dest_mesh)
             * 
             * Parameters:
             *  - force_alloc (optional): bool. If true, forces the views to have separate
             *    physical memory, even if the mesh was already on the host. Defaults to false.
            */
            HostMesh create_host_mirror(bool force_alloc = false){
                HostMesh host_mirror;
                // Copy dims
                host_mirror.n_points = n_points;
                host_mirror.n_edges = n_edges;
                host_mirror.n_regions = n_regions;

                // Create host mirrors
                if(force_alloc){
                    host_mirror.point_coords = Kokkos::create_mirror(point_coords);
                    host_mirror.edge_to_point_ids = Kokkos::create_mirror(edge_to_point_ids);
                    host_mirror.region_to_point_ids = Kokkos::create_mirror(region_to_point_ids);
                } else {
                    host_mirror.point_coords = Kokkos::create_mirror_view(point_coords);
                    host_mirror.edge_to_point_ids = Kokkos::create_mirror_view(edge_to_point_ids);
                    host_mirror.region_to_point_ids = Kokkos::create_mirror_view(region_to_point_ids);
                }

                return host_mirror;
            }

            /**
             * Deep copies all mesh buffers to the provided destination.
             * 
             * Can also be done by calling deep copy on each buffer individually,
             * or only on the subset of buffers modified.
            */
            void deep_copy_all_to(Mesh dest){
                Kokkos::deep_copy(dest.point_coords, point_coords);
                Kokkos::deep_copy(dest.edge_to_point_ids, edge_to_point_ids);
                Kokkos::deep_copy(dest.region_to_point_ids, region_to_point_ids);
            }

            // Size accessors
            inline int edge_count(){return n_edges;}
            inline int region_count(){return n_regions;}
            inline pointID point_count(){return n_points;}

            /**
             * Loads a mesh from a file into both a host and device mesh. 
             * 
             * Parameters:
             * 
             * File format:
             *  The first line must be:
             *    npnt: <np> nseg: <ne> ntri: <nr>
             *  Where np, ne, and nr are the number of points, edges, and regions respectively.
             *  Next must follow np lines of:
             *    <pt_id>: <x_coord> <y_coord>
             *  Where x_coord and y_coord are floating point strings parsable by strtof(). 
             *  Next must follow ne lines of:
             *    <edge_id>: <start_pt_id> <end_pt_id>
             *  Next must follow nr lines of:
             *    <reg_id>: <pt1_id> <pt2_id> <pt3_id>
             *  Lines after can have any contents.
             *  Left-hand side ID's should be in ascending order.
             *  (Anything in <> is replaced with its value- the file does not include angle brackets)  
            */
    };
      // Create alias for convenience
    typedef Mesh<Kokkos::DefaultHostExecutionSpace> HostMesh;
    typedef Mesh<Kokkos::DefaultExecutionSpace> DeviceMesh;

    void load_meshes_from_grd_file(std::string fname, DeviceMesh& device_mesh, HostMesh& host_mesh);
}

#endif