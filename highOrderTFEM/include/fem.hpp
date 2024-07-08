#ifndef highOrderTFEM_fem_hpp
#define highOrderTFEM_fem_hpp

#include <string>
#include <fstream>

#include <Kokkos_Core.hpp>
#include "mesh.hpp"
#include "type_magic.hpp"

namespace TFEM
{

    namespace SolverImpl
    {
        struct ElementContributionFunctor;
    }
    class SolutionWriter;

    class Solver
    {
        friend class SolutionWriter;
        friend class SolverImpl::ElementContributionFunctor;

    protected:
        // Stores 1/diagonals for the lumped diagonal mass matrix.
        using InvMassMatrix = Kokkos::View<double *, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
        using ConstInvMassMatrix = constify_view_t<InvMassMatrix>;
        InvMassMatrix point_mass_inv;
        ConstInvMassMatrix point_mass_inv_readonly;

        // Mesh and coloring
        DeviceMesh mesh;
        MeshColorMap element_coloring;

        // Parameters
        double dt;
        double k;
        int n_total_steps;


    public: 
        // Internal step simulation functions. Essentially called in order.
        // I would like very much for these to be private/protected, but
        // NVIDIA doesn't let device lambdas be in a private access space anywhere.
        /**
         * With the step that just completed stored in current_point_weights,
         * prepares things so that the new points can be overwritten.
         */
        void prepare_next_step();
        void compute_step();
        void fix_boundary();


        // Weight buffer
        using PointWeightBuffer = Kokkos::View<double *, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
        using ConstPointWeightBuffer = constify_view_t<PointWeightBuffer>;
        PointWeightBuffer current_point_weights;
        PointWeightBuffer prev_point_weights;
        ConstPointWeightBuffer prev_point_weights_readonly;

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

        SolverImpl::ElementContributionFunctor create_element_contribution_functor();
    };

    /**
     * To prevent issues such as implicit capture of the "this" pointer, per-element work is
     * placed in this functor where captures are more explicitly visible. 
    */
    namespace SolverImpl
    {
        // Functor called once with each element to update its contribution.
        // Can assume that each point and edge for each element are not being
        // touched by another thread.
        struct ElementContributionFunctor
        {
            Solver::PointWeightBuffer new_points;
            Solver::ConstPointWeightBuffer prev_points;
            Solver::ConstInvMassMatrix inv_mass;
            DeviceMesh mesh;
            double kdt;

            ElementContributionFunctor(Solver::PointWeightBuffer new_points,
                                       Solver::ConstPointWeightBuffer prev_points,
                                       Solver::ConstInvMassMatrix inv_mass,
                                       DeviceMesh mesh,
                                       double k, double dt) : new_points(new_points),
                                                              prev_points(prev_points),
                                                              inv_mass(inv_mass),
                                                              mesh(mesh),
                                                              kdt(k * dt) {}

            KOKKOS_INLINE_FUNCTION void operator()(Region element) const;
        };
    }

    class SolutionWriter
    {
    protected:
        std::ofstream out_file;
        DeviceMesh::HostMirrorMesh mesh;
        int slice_count;

    public:
        SolutionWriter(std::string fname, DeviceMesh::HostMirrorMesh mesh)
        {
            slice_count = 0;
            this->mesh = mesh;
            out_file = std::ofstream(fname);
            out_file << "{\"points\": [";
            for (pointID p = 0; p < mesh.point_count(); p++)
            {
                if (p > 0)
                {
                    out_file << ", ";
                }
                out_file << "[" << mesh.points(p)[0] << ", " << mesh.points(p)[1] << "]";
            }
            out_file << "],\n\"slices\":[";
        }

        template <class ViewType>
        void add_slice(ViewType view)
        {
            static_assert(Kokkos::is_view_v<ViewType>, "ViewType must be view");
            static_assert(ViewType::Rank == 1, "Points are arranged as a flat grid");
            static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename ViewType::memory_space>::accessible, "View must be accessible from host");
            assert(view.extent(0) == mesh.point_count());

            if (slice_count > 0)
            {
                out_file << ",";
            }
            slice_count++;

            out_file << std::endl
                     << "[";
            for (pointID p = 0; p < mesh.point_count(); p++)
            {
                if (p > 0)
                {
                    out_file << ", ";
                }
                out_file << view(p);
            }
            out_file << "]";
        }

        ~SolutionWriter()
        {
            // need to write out end cap to file
            out_file << "]}";
        }
    };
}

#endif