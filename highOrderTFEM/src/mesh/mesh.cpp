#include "mesh.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

using namespace TFEM;
using namespace std;

void TFEM::load_meshes_from_grd_file(string fname, DeviceMesh &device_mesh, DeviceMesh::HostMirrorMesh &host_mesh)
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
}