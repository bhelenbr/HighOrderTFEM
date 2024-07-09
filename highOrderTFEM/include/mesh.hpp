/**
 * Triangular mesh datastructure, including file IO and copying to/from device.
 */
#ifndef highOrderTFEM_mesh_hpp
#define highOrderTFEM_mesh_hpp

#include <Kokkos_Core.hpp>
#include <Kokkos_StaticCrsGraph.hpp> // for storing boundary edges
#include <string>
#include <type_traits>

namespace TFEM
{

    typedef int pointID; // in case this needs to be upgraded later, and for clear code semantics

    /**
     * Arrays have some trick device semantics (they are like pointers)
     * and multi-dimensional views are inconvenient for calling functions on single
     * items. Therefore the Point, Edge, and Region classes allow for easy array-like
     * usage while still being copyable.
     */
    struct Point
    {
    private:
        double coords[2];

    public:
        KOKKOS_INLINE_FUNCTION double &operator[](int i)
        {
            return coords[i];
        }
    };

    struct Edge
    {
    private:
        pointID points[2];

    public:
        KOKKOS_INLINE_FUNCTION pointID &operator[](int i)
        {
            return points[i];
        }
    };

    struct Region
    {
    private:
        pointID points[3];

    public:
        KOKKOS_INLINE_FUNCTION pointID &operator[](int i)
        {
            return points[i];
        }
    };

    /**
     * Handles the input mesh file as a collection of views. Since different view types may be required
     * (in particular for different execution spaces), template off of them.
     */
    template <class PointView, class EdgeView, class RegionView>
    class Mesh
    {
        // Assert that each view is indeed a view with correct type, and that memory spaces
        // line up. Don't care so much about other properties, which can be left flexible.
        // There is probably a better way of doing this templating/checking.
        static_assert(Kokkos::is_view_v<PointView>, "PointView must be kokkos view");
        static_assert(Kokkos::is_view_v<EdgeView>, "EdgeView must be kokkos view");
        static_assert(Kokkos::is_view_v<RegionView>, "RegionView must be kokkos view");
        using MemSpace = typename PointView::memory_space;
        static_assert(Kokkos::SpaceAccessibility<MemSpace, typename EdgeView::memory_space>::assignable);
        static_assert(Kokkos::SpaceAccessibility<MemSpace, typename RegionView::memory_space>::assignable);
        static_assert(std::is_same_v<typename PointView::data_type, Point *>, "PointView must be array of points");
        static_assert(std::is_same_v<typename EdgeView::data_type, Edge *>, "EdgeView must be array of edges");
        static_assert(std::is_same_v<typename RegionView::data_type, Region *>, "RegionView must be array of regions");

        // By default, different specializations of the same class don't have access to each other's private
        // members. We need this access for setting up device copies etc.
        template <class X, class Y, class Z>
        friend class Mesh;

    public:
        // Create a host mirror specialization for each specialization.
        typedef Mesh<typename PointView::HostMirror, typename EdgeView::HostMirror, typename RegionView::HostMirror> HostMirrorMesh;

    protected:
        pointID n_points;
        int n_edges;
        int n_regions;

    public:
        // Main buffers
        PointView points;
        EdgeView edges;
        RegionView regions;

        // Use CSR rowmap format to store IDs of boundary edges
        using BoundaryEdgeMap = Kokkos::StaticCrsGraph<int, typename EdgeView::execution_space>;
        BoundaryEdgeMap boundary_edges;

        /**
         * Sometimes need to leave an uninitialized mesh for later initialization
         */
        Mesh() = default;

        /**
         * Creates a mesh of given dimensions.
         */
        Mesh(int n_points, int n_edges, int n_regions)
            : n_points(n_points),
              n_edges(n_edges),
              n_regions(n_regions),
              points("mesh_points", n_points),
              edges("mesh_edges", n_edges),
              regions("mesh_regions", n_regions),
              boundary_edges()
        {
            // initializer list took care of most of it.
            // The boundary edges are a bit awkward: they have
            // strange initialization patterns, but are found at
            // the end of the file, so they must be initialized
            // specially.
        }

        /**
         * Creates a copy of the mesh with all primary views accessible from the host. Does not
         * copy or initialize the edge boundaries, since they have some initialization weirdness.
         *
         * Does NOT perform a deep copy- do manually or call src_mesh.deep_copy_all_to(dest_mesh)
         *
         * Parameters:
         *  - force_alloc (optional): bool. If true, forces the views to have separate
         *    physical memory, even if the mesh was already on the host. Defaults to false.
         */
        HostMirrorMesh create_host_mirror(bool force_alloc = false)
        {
            HostMirrorMesh host_mirror;
            // Copy dims
            host_mirror.n_points = n_points;
            host_mirror.n_edges = n_edges;
            host_mirror.n_regions = n_regions;

            // Create host mirrors
            if (force_alloc)
            {
                host_mirror.points = Kokkos::create_mirror(points);
                host_mirror.edges = Kokkos::create_mirror(edges);
                host_mirror.regions = Kokkos::create_mirror(regions);
            }
            else
            {
                host_mirror.points = Kokkos::create_mirror_view(points);
                host_mirror.edges = Kokkos::create_mirror_view(edges);
                host_mirror.regions = Kokkos::create_mirror_view(regions);
            }

            return host_mirror;
        }

        /**
         * Deep copies all mesh buffers to the provided destination (types permitting)
         *
         * Can also be done by calling deep copy on each buffer individually,
         * or only on the subset of buffers modified.
         */
        template <class X, class Y, class Z>
        void deep_copy_all_to(Mesh<X, Y, Z> dest)
        {
            Kokkos::deep_copy(dest.points, points);
            Kokkos::deep_copy(dest.edges, edges);
            Kokkos::deep_copy(dest.regions, regions);
        }

        // Size accessors
        inline int edge_count() { return n_edges; }
        inline int region_count() { return n_regions; }
        inline pointID point_count() { return n_points; }
    };

    // Create alias for convenience
    template <class ExecSpace>
    using ExecSpaceMesh = TFEM::Mesh<Kokkos::View<Point *, ExecSpace>, Kokkos::View<Edge *, ExecSpace>, Kokkos::View<Region *, ExecSpace>>;

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
    void load_meshes_from_grd_file(std::string fname, DeviceMesh &device_mesh, DeviceMesh::HostMirrorMesh &host_mesh);

    /**
     * Runs a (non-minimal) coloring algorithm on the regions in mesh so that no two elements
     * in the same color share a point. All regions of the same color can be queries as a
     * contiguous subview using "color_view(color_index)".
     *
     * Right now regions are copied by value (to save an extra dereference) so their index is lost.
     */
    class MeshColorMap
    {
    protected:
        // color index is a (n_colors + 1)-entry view where indices belonging to a
        // color are anything in [color_index(color), color_index(color + 1))
        Kokkos::View<int *> color_index;
        Kokkos::View<int *>::HostMirror color_index_host;
        // Array of regions sorted to be color-contiguous. See the index
        Kokkos::View<Region *> color_members;
        int n_colors;

    public:
        MeshColorMap(DeviceMesh &mesh);

        // When I put kokkos parallel for loops in the constructor,
        // the compiler yells at me that the enclosing function doesn't
        // have an adress (on GPU). This is a workaround- don't call.
        void do_color(DeviceMesh &mesh);

        /**
         * Number of colors used
         */
        int color_count();

        /**
         * Number of regions in the given color
         */
        int member_count(int color);

        auto color_view(int color)
        {
            return Kokkos::subview(color_members, Kokkos::pair(color_index_host[color], color_index_host[color + 1]));
        }
    };
}

#endif