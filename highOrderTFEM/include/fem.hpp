#ifndef highOrderTFEM_fem_hpp
#define highOrderTFEM_fem_hpp

#include <string>
#include <fstream>

#include <Kokkos_Core.hpp>
#include "mesh.hpp"
#include "type_magic.hpp"
#include "analytical.hpp"

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
        // Stores 1/diagonals (diagonals^-1) for the lumped diagonal mass matrix.
        using InvMassMatrix = Kokkos::View<double *, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
        using ConstInvMassMatrix = constify_view_t<InvMassMatrix>;
        InvMassMatrix point_mass_inv;
        ConstInvMassMatrix point_mass_inv_readonly;

        // Mesh and coloring
        DeviceMesh mesh;
        MeshColorMap element_coloring;
        Analytical::ZeroBoundary<> boundary;

        // Parameters
        double dt;
        double k;
        int n_total_steps;

    public:
        // Internal step simulation functions. Essentially called in order.
        // I would like very much for these to be private/protected, but
        // NVIDIA doesn't let device lambdas be in a private access space anywhere.
        // See: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda-restrictions
        /**
         * INTENDED PRIVATE: DO NOT CALL. Instead see simulate_steps()
         *
         * Initialize the system mass matrix and its readonly counterpart
         */
        void setup_mass_matrix();
        /**
         * INTENDED PRIVATE: DO NOT CALL. Instead see simulate_steps()
         *
         * Initialze current_point_weights with the initial values.
         * Prev points may be left as garbage values.
         */
        void setup_initial_conditions();
        /**
         * INTENDED PRIVATE: DO NOT CALL. Instead see simulate_steps()
         *
         * With the step that just completed stored in current_point_weights,
         * prepares things so that the new points can be overwritten.
         */
        void prepare_next_step();
        /**
         * INTENDED PRIVATE: DO NOT CALL. Instead see simulate_steps()
         *
         * Execute the main body of a step.
         */
        void compute_step();

        /**
         * INTENDED PRIVATE: DO NOT CALL. Instead see simulate_steps()
         *
         * For the moment, it seems easier to let the main loop body calculate whatever it wants for
         * values determined by the boundary conditions, and then fix them later. This way the main
         * loop doesn't need to spend time disambiguating whether or not a value is a boundary value.
         */
        void fix_boundary();

        /**
         * INTENDED PRIVATE: DO NOT CALL. Instead see simulate_steps()
         *
         * Creates a functor that can be called to process each element.
         */
        SolverImpl::ElementContributionFunctor create_element_contribution_functor();

    public:
        // This stuff was intended to be public, rather than being forced to make it accessible to
        // the nvidia compiler.

        // Weight buffers for storing state
        using PointWeightBuffer = Kokkos::View<double *, Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
        using ConstPointWeightBuffer = constify_view_t<PointWeightBuffer>;
        PointWeightBuffer current_point_weights;
        PointWeightBuffer prev_point_weights;
        ConstPointWeightBuffer prev_point_weights_readonly;

        Solver(DeviceMesh, MeshColorMap, Analytical::ZeroBoundary<>, double timestep, double k);

        /**
         * Runs the next n steps of the simulation
         */
        void simulate_steps(int n_steps);

        /**
         * Returns the mean squared pointwise error of the current timestep agaisnt the
         * original analytic solution.
         *
         * Note that this also measures error at the boundary, which may result in counterintuitive scaling.
         * Fix this later.
         */
        double measure_error();
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
                                       double k, double dt)
                : new_points(new_points),
                  prev_points(prev_points),
                  inv_mass(inv_mass),
                  mesh(mesh),
                  kdt(k * dt)
            { // Pretty much just the initializer list
            }

            /**
             * Adds the partial contributions of an element to all pertinent basis functions.
             *
             * TODO: put the magic here!
             */
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
            // need to write out end cap to file, which will close on its own
            // after it goes out of scope
            out_file << "]}";
        }
    };
}

#endif