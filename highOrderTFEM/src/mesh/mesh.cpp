#include "mesh.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <utility> // pair, etc

using namespace TFEM;
using namespace std;

std::default_random_engine rnd{std::random_device{}()};
uniform_real_distribution<double> dist(-1.0, 1.0);
// Reasonable heuristic for unit circle random. Not quite uniform, since we remap the corners.
std::pair<double, double> unit_circle_almost_random()
{

    double x = dist(rnd);
    double y = dist(rnd);

    // this will deform the distribution a little bit but unimportant for our use case.
    double rad = sqrt(pow(x, 2) + pow(y, 2));
    if (rad > 1.0)
    {
        double rescale = dist(rnd) / rad;
        x = x * rescale;
        y = y * rescale;
    }
    return std::make_pair(x, y);
}

void TFEM::load_meshes_from_grd_file(string fname, DeviceMesh &device_mesh, DeviceMesh::HostMirrorMesh &host_mesh, bool fuzz)
{
    ifstream input_file(fname);
    stringstream line_stream;
    // Read first line, get sizes, then initialize device view, then create host mirror. Read into
    // host mirror, then deep copy it back to the device view
    string cur_line;
    string string_buff;
    int line_no = 0;

    pointID n_points;
    int n_edges;
    int n_regions;
    double fuzz_radius = 0.0;

    // Read header
    getline(input_file, cur_line);
    line_no++;
    line_stream = stringstream(cur_line);
    line_stream >> string_buff;
    assert(string_buff == "npnt:");
    line_stream >> n_points;
    line_stream >> string_buff;
    assert(string_buff == "nseg:");
    line_stream >> n_edges;
    line_stream >> string_buff;
    assert(string_buff == "ntri:");
    line_stream >> n_regions;

    // if fuzz, assume a square grid on [-1, 1]^2 and calculate a safe fuzzing radius
    if (fuzz)
    {
        double grid_square_size = 2.0 / (sqrt(n_points) - 1);
        fuzz_radius = grid_square_size / 4; // could go up to sqrt(2)/4, but no need
    }

    // Create the meshes!
    device_mesh = DeviceMesh(n_points, n_edges, n_regions);
    host_mesh = device_mesh.create_host_mirror();

    // Read into host mesh
    // point coords
    for (pointID p_id = 0; p_id < n_points; p_id++)
    {
        getline(input_file, cur_line);
        line_no++;
        line_stream = stringstream(cur_line);

        pointID read_point_id;
        line_stream >> read_point_id;
        if (read_point_id != p_id)
        {
            throw runtime_error((string("Found unexpected / out-of-order ID at line ") + to_string(line_no)) + ": " + to_string(read_point_id));
        }
        // flush out the ":"
        line_stream >> string_buff;

        // Read point
        line_stream >> host_mesh.points(p_id)[0];
        line_stream >> host_mesh.points(p_id)[1];
    }
    // edge ID's
    for (int e_id = 0; e_id < n_edges; e_id++)
    {
        getline(input_file, cur_line);
        line_no++;
        line_stream = stringstream(cur_line);

        int read_edge_id;
        line_stream >> read_edge_id;
        if (read_edge_id != e_id)
        {
            throw runtime_error((string("Found unexpected / out-of-order ID at line ") + to_string(line_no)) + ": " + to_string(read_edge_id));
        }
        // flush out the ":"
        line_stream >> string_buff;
        line_stream >> host_mesh.edges(e_id)[0] >> host_mesh.edges(e_id)[1];
    }
    // region ID's
    for (int r_id = 0; r_id < n_regions; r_id++)
    {
        getline(input_file, cur_line);
        line_no++;
        line_stream = stringstream(cur_line);

        int read_region_id;
        line_stream >> read_region_id;
        if (read_region_id != r_id)
        {
            throw runtime_error((string("Found unexpected / out-of-order ID at line ") + to_string(line_no)) + ": " + to_string(read_region_id));
        }
        // flush out the ":"
        line_stream >> string_buff;
        line_stream >> host_mesh.regions(r_id)[0] >> host_mesh.regions(r_id)[1] >> host_mesh.regions(r_id)[2];
    }
    // copy over to device
    host_mesh.deep_copy_all_to(device_mesh);

    // Special handling of boundary edges
    getline(input_file, cur_line);
    line_no++;
    line_stream = stringstream(cur_line);
    // line should read nebd: <n_boundary_segments>
    int n_boundary_segments;
    line_stream >> string_buff >> n_boundary_segments;
    std::vector<std::vector<int>> boundary_segments(n_boundary_segments);
    // For each boundary segment, we expect two header lines followed by segment edges, i.e.
    // idnum: <i>
    // number: <n_edges_in_segment>
    // 0: <edge_1_id>
    // 1: <edge_2_id>
    // ...
    for (int seg = 0; seg < n_boundary_segments; seg++)
    {
        getline(input_file, cur_line);
        line_no++;
        // Don't care about first header right now so skip to next
        getline(input_file, cur_line);
        line_no++;
        line_stream = stringstream(cur_line);
        int n_edges_in_segment;
        line_stream >> string_buff >> n_edges_in_segment;
        for (int i = 0; i < n_edges_in_segment; i++)
        {
            getline(input_file, cur_line);
            line_no++;
            line_stream = stringstream(cur_line);

            int edge_id;
            line_stream >> string_buff >> edge_id;
            boundary_segments[seg].push_back(edge_id);
        }
    }

    // Construct the graph
    device_mesh.boundary_edges = Kokkos::create_staticcrsgraph<DeviceMesh::BoundaryEdgeMap>("Boundary edge segments", boundary_segments);
    host_mesh.boundary_edges = Kokkos::create_staticcrsgraph<DeviceMesh::HostMirrorMesh::BoundaryEdgeMap>("Boundary edge segments", boundary_segments);

    // track which points are boundaries and which are not. Fuzz points that are not, list points that are.
    host_mesh.boundary_points = DeviceMesh::HostMirrorMesh::BoundaryPointIndicator("Boundary edge flags", host_mesh.point_count());
    Kokkos::deep_copy(host_mesh.boundary_points, false);
    device_mesh.boundary_points = DeviceMesh::BoundaryPointIndicator("Boundary edge flags", host_mesh.point_count());

    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, host_mesh.boundary_edge_count()), KOKKOS_LAMBDA(int i) {
        Edge e = host_mesh.edges(host_mesh.boundary_edges.entries(i));
        host_mesh.boundary_points(e[0]) = true;
        host_mesh.boundary_points(e[0] = true); });

    host_mesh.n_boundary_points = 0;

    for (int pointID = 0; pointID < host_mesh.point_count(); pointID++)
    {
        if (host_mesh.boundary_points(pointID))
        {
            host_mesh.n_boundary_points++;
        }
        else if (fuzz)
        {
            // fuzz
            Point &point = host_mesh.points(pointID);
            auto displacement = unit_circle_almost_random();
            point[0] += fuzz_radius * displacement.first;
            point[1] += fuzz_radius * displacement.second;
        }
    }
    // Deep copy point displacements
    Kokkos::deep_copy(device_mesh.points, host_mesh.points);
    // Copy point boundary keys
    Kokkos::deep_copy(device_mesh.boundary_points, host_mesh.boundary_points);
    device_mesh.n_boundary_points = host_mesh.n_boundary_points;
}