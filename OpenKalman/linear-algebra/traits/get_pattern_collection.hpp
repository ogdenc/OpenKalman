/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref get_pattern_collection function.
 */

#ifndef OPENKALMAN_GET_PATTERN_COLLECTION_HPP
#define OPENKALMAN_GET_PATTERN_COLLECTION_HPP

#include "coordinates/coordinates.hpp"
#include "linear-algebra/interfaces/interfaces-defined.hpp"
#include "linear-algebra/traits/get_mdspan.hpp"
#include "linear-algebra/traits/index_count.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<std::size_t i, typename Mdspan>
    static constexpr auto
    get_extent(const Mdspan& m)
    {
      constexpr auto ex = Mdspan::static_extent(i);
      if constexpr (ex == stdex::dynamic_extent)
        return m.extent(i);
      else
        return std::integral_constant<std::size_t, ex>{};
    }

    template<typename Mdspan, std::size_t...i>
    static constexpr auto
    get_pattern_collection_impl(const Mdspan& m, std::index_sequence<i...>)
    {
      return std::tuple {get_extent<i>(m)...};
    }

  }


  /**
   * \brief Get the \ref coordinates::pattern_collection associated with \ref indexible object T.
   */
#ifdef __cpp_concepts
  template<indexible T>
  constexpr coordinates::pattern_collection decltype(auto)
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr decltype(auto)
#endif
  get_pattern_collection(T&& t)
  {
    using Td = stdex::remove_cvref_t<T>;
    if constexpr (interface::get_pattern_collection_defined_for<Td>)
    {
      auto pat = stdex::invoke(interface::object_traits<Td>::get_pattern_collection, std::forward<T>(t));
      static_assert(not values::size_compares_with<coordinates::dimension_of<decltype(pat)>, index_count<T>, &stdex::is_neq>);
      return pat;
    }
    else
    {
      return detail::get_pattern_collection_impl(get_mdspan(t), std::make_index_sequence<index_count_v<T>>{});
    }
  }

}

#endif
