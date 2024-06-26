/**
 * Triangular mesh datastructure, including file IO and copying to/from device.
*/
#include <Kokkos_Core.hpp>
#include <string>
#include <type_traits>


namespace TFEM {

    typedef uint16_t pointID;

    struct Point{
        double x;
        double y;
    };
    struct Edge{
        pointID p1;
        pointID p2;
    };
    struct Region{
        pointID p1;
        pointID p2;
        pointID p3;
    };


    template <class PointView, class EdgeView, class RegionView>
    class Mesh {
        // Assert that each view is indeed a view with correct type, and that memory spaces
        // line up. Don't care so much about other properties, which can be left flexible.
        static_assert(Kokkos::is_view_v<PointView>, "PointView must be kokkos view");
        static_assert(Kokkos::is_view_v<EdgeView>, "EdgeView must be kokkos view");
        static_assert(Kokkos::is_view_v<RegionView>, "RegionView must be kokkos view");
        static_assert(std::is_same_v<typename PointView::data_type, Point*>, "PointView must be array of points");
        static_assert(std::is_same_v<typename EdgeView::data_type, Edge*>, "EdgeView must be array of edges");
        static_assert(std::is_same_v<typename RegionView::data_type, Region*>, "RegionView must be array of regions");

        // We are in the arcane depths of C++ now: can't access protected accross memory spaces
        // since a different specialization is technically a different class (?)
        // Make friend declaration
        template <class X, class Y, class Z> friend class Mesh;

        // Create type aliases for clarity and for easy size tweaks (i.e. upgrade to long double)
        public:
            typedef Mesh<typename PointView::HostMirror, typename EdgeView::HostMirror, typename RegionView::HostMirror> HostMirrorMesh;

        protected:
            pointID n_points;
            int n_edges;
            int n_regions;

        public:
            PointView points;
            EdgeView  edges;
            RegionView regions;
            
            /**
             * Sometimes need to leave an uninitialized mesh.
            */
            Mesh() = default;

            /**
             * Creates a mesh of given dimensions.
            */
            Mesh(int n_points, int n_edges, int n_regions) : 
                n_points(n_points), n_edges(n_edges), n_regions(n_regions),
                points("mesh_points", n_points),
                edges("mesh_edges", n_edges),
                regions("mesh_regions", n_regions)
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
            HostMirrorMesh create_host_mirror(bool force_alloc = false){
                HostMirrorMesh host_mirror;
                // Copy dims
                host_mirror.n_points = n_points;
                host_mirror.n_edges = n_edges;
                host_mirror.n_regions = n_regions;

                // Create host mirrors
                if(force_alloc){
                    host_mirror.points = Kokkos::create_mirror(points);
                    host_mirror.edges = Kokkos::create_mirror(edges);
                    host_mirror.regions = Kokkos::create_mirror(regions);
                } else {
                    host_mirror.points = Kokkos::create_mirror_view(points);
                    host_mirror.edges = Kokkos::create_mirror_view(edges);
                    host_mirror.regions = Kokkos::create_mirror_view(regions);
                }

                return host_mirror;
            }

            /**
             * Deep copies all mesh buffers to the provided destination.
             * 
             * Can also be done by calling deep copy on each buffer individually,
             * or only on the subset of buffers modified.
            */
            template <class X, class Y, class Z>
            void deep_copy_all_to(Mesh<X, Y, Z> dest){
                Kokkos::deep_copy(dest.points, points);
                Kokkos::deep_copy(dest.edges, edges);
                Kokkos::deep_copy(dest.regions, regions);
            }

            // Size accessors
            inline int edge_count(){return n_edges;}
            inline int region_count(){return n_regions;}
            inline pointID point_count(){return n_points;}

    };

    // Create alias for convenience
    template<class ExecSpace> 
    using ExecSpaceMesh = TFEM::Mesh<Kokkos::View<Point*, ExecSpace>, Kokkos::View<Edge*, ExecSpace>, Kokkos::View<Region*, ExecSpace>>;

    typedef ExecSpaceMesh<Kokkos::DefaultExecutionSpace> DeviceMesh;

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
    void load_meshes_from_grd_file(std::string fname, DeviceMesh& device_mesh, DeviceMesh::HostMirrorMesh& host_mesh);
}