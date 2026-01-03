/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref patterns::get_pattern.
 */

#ifndef OPENKALMAN_COLLECTIONS_GET_PATTERN_HPP
#define OPENKALMAN_COLLECTIONS_GET_PATTERN_HPP

#include "collections/collections.hpp"
#include "patterns/concepts/pattern_collection.hpp"
#include "patterns/descriptors/Dimensions.hpp"
#include "patterns/descriptors/Any.hpp"

namespace OpenKalman::patterns
{
  /**
   * \brief Get a \ref pattern within a \ref pattern_collection
   * \details If index i is greater than the size of argument p, the function will return Dimensions<1> or 1.
   */
#ifdef __cpp_concepts
  template<pattern_collection P, values::index I>
#else
  template<typename P, typename I, std::enable_if_t<
    pattern_collection<P> and values::index<I>, int> = 0>
#endif
  constexpr decltype(auto)
  get_pattern(P&& p, I i)
  {
    if constexpr (not collections::sized<P> or values::size_compares_with<I, collections::size_of<P>, &stdex::is_lt>)
      return collections::get_element(std::forward<P>(p), i);
    else if constexpr (values::size_compares_with<I, collections::size_of<P>, &stdex::is_gteq>)
      return Dimensions<1>{};
    else if (i < collections::get_size(p))
      return Any {collections::get_element(std::forward<P>(p) | collections::views::all, i)};
    else
      return Any {1_uz};
  }


/**
 * \overload
 */
#ifdef __cpp_concepts
  template<std::size_t i, pattern_collection P>
#else
  template<std::size_t i, typename P, std::enable_if_t<pattern_collection<P>, int> = 0>
#endif
  constexpr decltype(auto)
  get_pattern(P&& p)
  {
    return get_pattern(std::forward<P>(p), std::integral_constant<std::size_t, i>{});
  }


}


#endif
