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

    // Type alias in case we need to change the type later, and to make the code more readable.
    typedef int pointID;

    // Arrays have some tricky device copy behavior (since the type behaves like a pointer),
    // so these structs are to make bundling a single point/edge/region easier. This could
    // also be approached via a multidimensional view, with rows being the entries of the
    // same point, but this makes it more dificult to move single points around.

    /**
     * A struct containing the real-space coordinates of a point.
     * Access using array [ . ] notation.
     */
    struct Point
    {
    private:
        double coords[2];

    public:
        /**
         * For i = 0 or 1, return the x or y coordinate of the point respectively.
         */
        KOKKOS_INLINE_FUNCTION double &operator[](int i)
        {
            return coords[i];
        }
    };

    /**
     * A struct containing the id's of the points at the end of each edge. IDs are
     * in the context of the original mesh this edge belongs to.
     *
     * Access using array [ . ] notation.
     */
    struct Edge
    {
    private:
        pointID points[2];

    public:
        /**
         * For i = 0 or 1, return the ID of the point at the corresponding endpoint of this edge.
         */
        KOKKOS_INLINE_FUNCTION pointID &operator[](int i)
        {
            return points[i];
        }
    };

    /**
     * A struct containing the id's of the points that make up the vertices of a triangular
     * mesh region. IDs are in the context of the original mesh.
     *
     * Access using array [ . ] notation.
     */
    struct Region
    {
    private:
        pointID points[3];

    public:
        /**
         * For i in [0, 2], return the ID of the point at the corresponding vertex.
         */
        KOKKOS_INLINE_FUNCTION pointID &operator[](int i)
        {
            return points[i];
        }
    };

    /**
     * Represents a mesh as a list of points, edges, and regions. Also contains information
     * on the boundary edges (grouped in segments) and boundary points. Typically, create
     * this mesh by calling "load_meshes_from_grd_file".
     *
     * Templates on various view types to accomodate different execution spaces and memory
     * access patterns. All views should be accessible from the same space.
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
        static_assert(Kokkos::SpaceAccessibility<MemSpace, typename EdgeView::memory_space>::accessible);
        static_assert(Kokkos::SpaceAccessibility<MemSpace, typename RegionView::memory_space>::accessible);
        static_assert(std::is_same_v<typename PointView::data_type, Point *>, "PointView must be array of points");
        static_assert(std::is_same_v<typename EdgeView::data_type, Edge *>, "EdgeView must be array of edges");
        static_assert(std::is_same_v<typename RegionView::data_type, Region *>, "RegionView must be array of regions");

        // By default, different specializations of the same class don't have access to each other's private
        // members. We need this access for setting up device copies etc.
        template <class X, class Y, class Z>
        friend class Mesh;

        // Mesh size counts
        int n_points;
        int n_edges;
        int n_regions;

    public:
        // Create a host mirror specialization for each specialization.
        typedef Mesh<typename PointView::HostMirror, typename EdgeView::HostMirror, typename RegionView::HostMirror> HostMirrorMesh;

        // Main buffers
        PointView points;
        EdgeView edges;
        RegionView regions;
        int n_boundary_points;


        /**
         * Use a CSR-style datastructure to create a list of boundary edges, grouped into segments
         * as identified in the original mesh.
         *
         * If the segment grouping is uninmportant, a simple list of boundary edges is stored on
         * boundary_edges.edges
         */
        using BoundaryEdgeMap = Kokkos::StaticCrsGraph<int, typename EdgeView::execution_space>;
        BoundaryEdgeMap boundary_edges;

        /**
         * An array of boolean flags, on for each point in the mesh. The flag is true iff
         * the point lies on a boundary edge.
         */
        using BoundaryPointIndicator = Kokkos::View<bool *, typename EdgeView::execution_space>;
        BoundaryPointIndicator boundary_points;

        /**
         * Sometimes need to leave an uninitialized mesh for later initialization
         */
        Mesh() = default;

        /**
         * Creates a mesh of given dimensions. Does not do much further initialization:
         * most of this is done externally in "load_meshes_from_grd_file"
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
         * copy or initialize the edge boundaries or points, since they have some initialization weirdness.
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
        inline int boundary_edge_count() { return boundary_edges.entries.extent(0); }
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
     *  Afterwards we parse the boundary segments and edges:
     *    nebd: <num_edge_segments>
     *  For each boundary segment, parse the edges:
     *    idnum: <segment_id>
     *    number: <n_edges_in_segment>
     *  Followed by n_edges_in_segment lines of
     *    <segment_edge_index>: <edge_id>
     *  Lines after can have any contents.
     *  Left-hand side ID's should be in ascending order.
     *  (Anything in <> is replaced with its value- the file does not include angle brackets)
     */
    void load_meshes_from_grd_file(std::string fname, DeviceMesh &device_mesh, DeviceMesh::HostMirrorMesh &host_mesh, bool fuzz = false);

    /**
     * Runs a (non-minimal) coloring algorithm on the regions in mesh so that no two elements
     * in the same color share a point. All regions of the same color can be queries as a
     * contiguous subview using "color_member_regions(color_index)".
     *
     * Right now regions are copied by value (to save an extra dereference) so their index is lost.
     */
    class MeshColorMap
    {
    protected:
        // color index is a (n_colors + 1)-entry view where indices belonging to a
        // color are anything in [color_index(color), color_index(color + 1))
        Kokkos::View<const int *> color_index;
        Kokkos::View<const int *>::HostMirror color_index_host;
        // Array of regions sorted to be color-contiguous. See the index
        Kokkos::View<const Region *> color_members;

        // original mesh ID's corresponding to each region
        Kokkos::View<const int *> color_ids;
        Kokkos::View<const int *>::HostMirror color_ids_host;
        int n_colors;

        auto color_endpoints(int color)
        {
            return Kokkos::pair(color_index_host[color], color_index_host[color + 1]);
        }

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

        /**
         * Returns a device-accessible contiguous subview of the regions in the indicated color
         */
        auto color_member_regions(int color)
        {
            return Kokkos::subview(color_members, color_endpoints(color));
        }

        /**
         * Returns a device-accessible subview containing the ID's of the regions of the indicated color
         */
        auto color_member_ids(int color)
        {
            return Kokkos::subview(color_ids, color_endpoints(color));
        }

        /**
         * Returns a host-accessible subview containing the ID's of the regions of the indicated color
         */
        auto color_member_ids_host(int color)
        {
            return Kokkos::subview(color_ids_host, color_endpoints(color));
        }
    };

    void validate_mesh_coloring(typename DeviceMesh::HostMirrorMesh &mesh, MeshColorMap &coloring);

}

#endif