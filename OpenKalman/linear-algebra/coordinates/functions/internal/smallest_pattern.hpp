/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref internal::smallest_pattern function.
 */

#ifndef OPENKALMAN_SMALLEST_PATTERN_HPP
#define OPENKALMAN_SMALLEST_PATTERN_HPP

#include <algorithm>
#include "collections/collections.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/pattern_collection.hpp"

namespace OpenKalman::coordinates::internal
{
  namespace details
  {
    template<std::size_t i = 0, typename P,
      typename MI = std::integral_constant<std::size_t, 0>,
      typename Min = std::integral_constant<std::size_t, std::numeric_limits<std::size_t>::max()>>
    constexpr auto
    smallest_pattern_fixed(const P& p, MI mi = {}, Min min = {})
    {
      if constexpr (i < collections::size_of_v<P>)
      {
        auto ix = std::integral_constant<std::size_t, i>{};
        auto p_i = get_dimension(collections::get(p, ix));
        auto is_new_min = values::operation(std::less{}, p_i, min);
        auto select = [](bool q, std::size_t curr, std::size_t old){ return q ? curr : old; };
        return smallest_pattern_fixed<i + 1>(
          p,
          values::operation(select, is_new_min, ix, mi),
          values::operation(select, is_new_min, p_i, min));
      }
      else
      {
        return mi;
      }
    }
  }


  /**
   * \internal
   * \brief Return an index to the smallest \ref pattern within a \ref pattern_collection
   * \details If there are multiple candidates, it will choose the earliest-listed one.
   * \returns an index to the minimum value
   */
#ifdef __cpp_concepts
  template<pattern_collection P> requires collections::sized<P> and
    (collections::size_of_v<P> == dynamic_size or collections::size_of_v<P> > 0_uz)
  constexpr values::index auto
#else
  template<typename P, std::enable_if_t<pattern_collection<P> and collections::sized<P> and
    (collections::size_of<P>::value == dynamic_size or collections::size_of<P>::value > 0_uz), int> = 0>
  constexpr auto
#endif
  smallest_pattern(const P& p)
  {
    if constexpr (collections::size_of_v<P> == dynamic_size)
    {
      auto fn = [](auto a, auto b){ return get_dimension(a) < get_dimension(b); };
      decltype(auto) pr = collections::views::all(p);
#ifdef __cpp_lib_ranges
      auto it = std::ranges::min_element(pr, fn);
#else
      auto it = std::min_element(pr.begin(), pr.end(), fn);
#endif
      return static_cast<std::size_t>(std::distance(pr.begin(), it));
    }
    else
    {
      return details::smallest_pattern_fixed(p);
    }
  }

}

#endif
