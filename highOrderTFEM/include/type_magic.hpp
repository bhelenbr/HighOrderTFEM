#ifndef highOrderTFEM_type_magic_hpp
#define highOrderTFEM_type_magic_hpp

#include <Kokkos_Core.hpp>

namespace TFEM {
    /*
    * Get the const version of a view type. Maybe this exists within Kokkos, but I couldn't
    * find it.
    */
    template <typename> struct constify_view {};

    template <typename EntryT, typename... ViewArgs>
    struct constify_view<Kokkos::View<EntryT, ViewArgs...>>{
        using old_type = Kokkos::View<EntryT, ViewArgs...>;
        using type = Kokkos::View<typename old_type::const_data_type, ViewArgs...>;
    };

    template <typename ViewT>
    using constify_view_t = typename constify_view<ViewT>::type;

}

#endif