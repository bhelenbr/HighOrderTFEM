#include <Kokkos_StaticCrsGraph.hpp>
#include <KokkosGraph_Distance2Color.hpp>
#include <KokkosSparse_CrsMatrix.hpp>

#include "mesh.hpp"

// debug includes
#include <vector>
#include <iostream>

using namespace TFEM;

MeshColorMap::MeshColorMap(DeviceMesh &mesh)
{
    do_color(mesh);
}

void MeshColorMap::do_color(DeviceMesh &mesh)
{
    // Call the bipartite row coloring to assign different
    // colors to any element sharing a neighboring vertex,
    // as described at:
    // https://github.com/kokkos/kokkos-kernels/wiki/D2-Graph-Coloring#bipartite-graph-row-coloring

    using RowMapType = Kokkos::View<int *>;
    using IndexArrayType = Kokkos::View<pointID *>;
    using HandleType = KokkosKernels::Experimental::KokkosKernelsHandle<
        RowMapType::non_const_value_type,
        IndexArrayType::non_const_value_type,
        double,
        Kokkos::DefaultExecutionSpace,
        Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::DefaultExecutionSpace::memory_space>;

    // row entries view. Almost identical to the original elements-to-vertex view, except has one greater
    // extent and I don't want to fiddle with memory layout bugs, so we make a new array.
    int n_elements = mesh.region_count();
    RowMapType row_start_map("Row starts", n_elements + 1);
    IndexArrayType indices_array("Column indices", 3 * n_elements); // a region is a triangle of 3 points

    // Populate the array with the points to color on
    Kokkos::parallel_for(row_start_map.extent(0), KOKKOS_LAMBDA(int i) {
        if(i >= n_elements){
            // per design of the kokkos_kernels function, this must
            // store the extent of the indices array
            row_start_map(i) = indices_array.extent(0);
        } else {
            row_start_map(i) = 3 * i;
            Region points = mesh.regions(i);
            for(int j = 0; j < 3; j++){
                indices_array(3*i + j) = points[j];
            }
        } });

    // Call into the kernel
    HandleType handle;
    handle.create_distance2_graph_coloring_handle();
    KokkosGraph::Experimental::bipartite_color_rows(&handle, mesh.region_count(), mesh.point_count(), row_start_map, indices_array);

    int n_colors = handle.get_distance2_graph_coloring_handle()->get_num_colors();
    this->n_colors = n_colors; // We need to keep a local reference of n_colors due to lambda capture
    auto region_to_colors = handle.get_distance2_graph_coloring_handle()->get_vertex_colors();

    // Cleanup.
    handle.destroy_distance2_graph_coloring_handle();

    // Colors indexes by region and gives the color. We want to pick a color
    // and iterate over the regions, requiring some restructuring.
    // This can be done in two parallel passes over the original color array.
    // Keep views as separate from the object fields until the end to prevent
    // lambda capture of the "this" pointer.
    Kokkos::View<int *> color_counts("Color counts", n_colors);
    Kokkos::View<int *> color_index("Color index", n_colors + 1);
    Kokkos::View<Region *> color_members("Color members", region_to_colors.extent(0));
    Kokkos::View<int *> color_member_ids("Color member_ids", region_to_colors.extent(0));

    // Step 1: count how many items are in each color.
    // Atomic add should be a fairly safe function to use for 32-bit integers.
    Kokkos::parallel_for(mesh.region_count(), KOKKOS_LAMBDA(int i) {
        int color = region_to_colors(i) - 1;
        Kokkos::atomic_increment(&color_counts(color)); });

    // Step 2: cumulative sum to get start points.
    // A parallel for with 1 iteration is a bit awkward, but is a simple way
    // to run something in the device execution space.
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(int _) {
        for(int c = 0; c < n_colors; c++){
            color_index[c + 1] = color_index[c] + color_counts[c];
        } });

    // Step 3: If we reset the counts and do atomic_fetch_increment,
    // will will be able to tell which item within each color we are,
    // so we can index relative to the startpoint.
    Kokkos::deep_copy(color_counts, 0);
    Kokkos::parallel_for(mesh.region_count(), KOKKOS_LAMBDA(int i) {
        int color = region_to_colors(i) - 1; // colors start at 1 but our array starts at 0
        int place_ind = color_index(color) + Kokkos::atomic_fetch_add(&color_counts[color], 1);
        color_members(place_ind) = mesh.regions(i);
        color_member_ids(place_ind) = i; });

    // Now we should have a nice CSR-like structure for iterating over colors!
    // Just need to make the indexing available at the host:
    this->color_index = color_index;
    this->color_index_host = Kokkos::create_mirror_view(color_index);
    Kokkos::deep_copy(this->color_index_host, color_index);
    this->color_members = color_members;
    this->color_ids = color_member_ids;
    this->color_ids_host = Kokkos::create_mirror_view(color_member_ids);
    Kokkos::deep_copy(this->color_ids_host, color_member_ids);
}

int MeshColorMap::color_count()
{
    return n_colors;
}

int MeshColorMap::member_count(int color)
{
    return color_index_host(color + 1) - color_index_host(color);
}

/**
 * Runs a check and prints to the console that a coloring is a correct. A coloring is correct if
 * every region in the mesh has exactly one color, and no regions within a single color share a point.
 *
 * In a more mature project this would be part of a test suite instead.
 */
void TFEM::validate_mesh_coloring(typename DeviceMesh::HostMirrorMesh &mesh, MeshColorMap &coloring)
{
    using IntArray = Kokkos::View<int *, Kokkos::DefaultHostExecutionSpace>;

    // Go through all ID's and see if any have color colisions.
    IntArray region_colors("ID occurence count", mesh.region_count());
    Kokkos::deep_copy(region_colors, -1);

    std::cout << "Validating uniqueness of coloring..." << std::endl;
    for (int color = 0; color < coloring.color_count(); color++)
    {
        auto color_ids = coloring.color_member_ids_host(color);
        for (int i = 0; i < color_ids.extent(0); i++)
        {
            int id = color_ids(i);
            if (region_colors(id) > -1)
            {
                std::cout << "Region " << id << " already colored " << color_ids(id) << ", again colored " << color << std::endl;
            }
            else
            {
                region_colors(id) = color;
            }
        }
    }
    std::cout << "Done." << std::endl;

    // Check that every point is touched no more than once per color
    IntArray point_regions("point color validation", mesh.point_count());

    std::cout << "Validating nonadjacency of same-color regions..." << std::endl;
    for (int color = 0; color < coloring.color_count(); color++)
    {
        Kokkos::deep_copy(point_regions, -1);
        auto color_ids = coloring.color_member_ids_host(color);
        for (int i = 0; i < color_ids.extent(0); i++)
        {
            int id = color_ids(i);
            Region r = mesh.regions(id);
            for (int j = 0; j < 3; j++)
            {
                pointID p = r[j];
                if (point_regions(p) > -1)
                {
                    std::cout << "Same-color (" << color << ") conflict between regions" << point_regions(p) << " and " << id << " at point " << p << std::endl;
                }
                else
                {
                    point_regions(p) = id;
                }
            }
        }
    }
    std::cout << "Done" << std::endl;
}
