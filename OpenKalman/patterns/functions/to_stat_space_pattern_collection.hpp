/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref patterns::to_stat_space_pattern_collection.
 */

#ifndef OPENKALMAN_TO_STAT_SPACE_PATTERN_COLLECTION_HPP
#define OPENKALMAN_TO_STAT_SPACE_PATTERN_COLLECTION_HPP

#include "collections/collections.hpp"
#include "patterns/concepts/pattern_collection.hpp"
#include "patterns/concepts/euclidean_pattern_collection.hpp"
#include "patterns/functions/to_extents.hpp"

namespace OpenKalman::patterns
{
  /**
   * \brief Convert one \ref pattern_collection to another corresponding to the result of a \ref to_stat_space transformation.
   * \details This will potentially alter the dimension of the first pattern in the collection without changing any of the others.
   */
#ifdef __cpp_lib_constexpr_vector
  template<pattern_collection P>
  constexpr euclidean_pattern_collection auto
#else
  template<typename P, std::enable_if_t<pattern_collection<P>, int> = 0>
  constexpr auto
#endif
  to_stat_space_pattern_collection(P&& p)
  {
    if constexpr (euclidean_pattern_collection<P>)
      return std::forward<P>(p);
    else
      return to_extents(collections::views::concat(
        std::array {get_stat_dimension(get_pattern<0>(p))},
        collections::views::slice(std::forward<P>(p), std::integral_constant<std::size_t, 1>{})));
  }

}


#endif
