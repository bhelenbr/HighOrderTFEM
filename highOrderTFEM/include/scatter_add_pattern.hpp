#ifndef highOrderTFEM_scatter_add_pattern_hpp
#define highOrderTFEM_scatter_add_pattern_hpp

#include <Kokkos_Core.hpp>

#include "mesh.hpp"

namespace TFEM
{
    /**
     * Example signature for a scatter add pattern. No need to inherit directly, as long as
     * the signature is met. The outer template (ContributeParams) must be concrete,
     * but the inner template must remain a template.
     *
     * Once instantiated, functors should call the instances contribute method when
     * they need to do their op.
     */
    // class ScatterAddPattern
    // {
    //     template <typename WorkerFunctor>
    //     void distribute_work(WorkerFunctor functor);

    //     static KOKKOS_INLINE_FUNCTION void contribute(Arg1 arg1, Arg2 arg2, ...);
    // }

    class AtomicElementScatterAdd
    {
    private:
        DeviceMesh mesh;

    public:
        AtomicElementScatterAdd(DeviceMesh mesh)
        {
            this->mesh = mesh;
        }

        template <typename WorkerFunctor>
        void distribute_work(WorkerFunctor functor)
        {
            auto mesh = this->mesh;
            Kokkos::parallel_for(mesh.region_count(), KOKKOS_LAMBDA(int element_id) {
                Region element = mesh.regions(element_id);
                functor(element); });
        }

        static KOKKOS_INLINE_FUNCTION void contribute(double *dest, double contribution)
        {
            Kokkos::atomic_add(dest, contribution);
        }
    };

    class ColoredElementScatterAdd
    {
    private:
        MeshColorMap coloring;

    public:
        ColoredElementScatterAdd(MeshColorMap coloring) : coloring(coloring)
        {
        }

        template <typename WorkerFunctor>
        void distribute_work(WorkerFunctor functor)
        {
            for (int color = 0; color < coloring.color_count(); color++)
            {
                auto elements = coloring.color_member_regions(color);

                Kokkos::parallel_for(elements.extent(0), KOKKOS_LAMBDA(int i) {
                    Region element = elements(i);
                    functor(element); });
                Kokkos::fence();
            }
        }

        static KOKKOS_INLINE_FUNCTION void contribute(double *dest, double contribution)
        {
            *dest += contribution;
        }
    };
}

#endif // Include guard