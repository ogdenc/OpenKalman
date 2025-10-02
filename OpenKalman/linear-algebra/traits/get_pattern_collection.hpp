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
#include "linear-algebra/interfaces/object-traits-defined.hpp"
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
      if constexpr (ex == stdcompat::dynamic_extent)
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
  constexpr coordinates::pattern auto
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto
#endif
  get_pattern_collection(T&& t)
  {
    using Td = std::remove_reference_t<T>;
    if constexpr (interface::get_pattern_collection_defined_for<Td>)
      return stdcompat::invoke(interface::indexible_object_traits<Td>::get_pattern_collection, t);
    else
      return detail::get_pattern_collection_impl(get_mdspan(t), std::make_index_sequence<index_count_v<T>>{});
  }

}

#endif
