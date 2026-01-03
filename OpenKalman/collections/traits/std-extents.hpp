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
 * \internal
 * \file
 * \brief Interface that makes std::extents into a \ref collection.
 */

#ifndef OPENKALMAN_STD_EXTENTS_HPP
#define OPENKALMAN_STD_EXTENTS_HPP

#include "values/values.hpp"

namespace OpenKalman::collections
{
#ifdef __cpp_concepts
  template<typename IndexType, std::size_t...Extents, values::index I>
    requires (not values::fixed_value_compares_with<I, sizeof...(Extents), &std::is_gteq>)
  constexpr values::size auto
#else
  template<typename IndexType, std::size_t...Extents, typename I, std::enable_if_t<values::index<I> and
    (not values::fixed_value_compares_with<I, sizeof...(Extents), &stdex::is_gteq>), int> = 0>
  constexpr auto
#endif
  get_element(const stdex::extents<IndexType, Extents...>& e, I i) noexcept
  {
    if constexpr (values::fixed<I>)
    {
      constexpr std::size_t ix = values::fixed_value_of_v<I>;
      constexpr std::size_t s = std::decay_t<decltype(e)>::static_extent(ix);
      if constexpr (s == stdex::dynamic_extent) return e.extent(ix);
      else return std::integral_constant<std::size_t, s>{};
    }
    else
    {
      return e.extent(i);
    }
  }


  template<typename IndexType, std::size_t...Extents>
#ifdef __cpp_concepts
  constexpr values::size auto
#else
  constexpr auto
#endif
  get_size(const stdex::extents<IndexType, Extents...>&)
  {
    return std::integral_constant<std::size_t, sizeof...(Extents)>{};
  };


}


#endif
