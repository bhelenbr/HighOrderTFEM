/**
 * Static utilities and type traits for dealing with Kokkos.
 */

#ifndef highOrderTFEM_type_magic_hpp
#define highOrderTFEM_type_magic_hpp

#include <Kokkos_Core.hpp>

namespace TFEM
{
    /*
     * Type trait to get the read-only view type corresponding to the given type.
     * 
     * This may already exist within Kokkos, but I couldn't find it.
     */
    template <typename>
    struct constify_view
    { // Do nothing for generic types, define behavior only for views.
    };

    template <typename EntryT, typename... ViewArgs>
    struct constify_view<Kokkos::View<EntryT, ViewArgs...>>
    {
        using old_type = Kokkos::View<EntryT, ViewArgs...>;
        using type = Kokkos::View<typename old_type::const_data_type, ViewArgs...>;
    };

    // Convienient alias to type constify_view_t<MyViewT> instead of constify_view<MyViewT>::type everywhere
    template <typename ViewT>
    using constify_view_t = typename constify_view<ViewT>::type;

}

#endif