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
 * \internal
 * \brief Definition for \ref to_diagonal_pattern_collection.
 */

#ifndef OPENKALMAN_TO_DIAGONAL_PATTERN_COLLECTION_HPP
#define OPENKALMAN_TO_DIAGONAL_PATTERN_COLLECTION_HPP

#include "collections/collections.hpp"
#include "patterns/concepts/pattern_collection.hpp"
#include "patterns/functions/get_pattern.hpp"

namespace OpenKalman::patterns
{
  /**
   * \brief Convert one \ref pattern_collection to another that is equivalent to duplicating the first index.
   * \details In the result, the pattern for ranks 0 and 1 will both be the pattern for rank 0 in the argument.
   */
#ifdef __cpp_concepts
  template<pattern_collection P>
  constexpr pattern_collection auto
#else
  template<typename P, std::enable_if_t<pattern_collection<P>, int> = 0>
  constexpr auto
#endif
  to_diagonal_pattern_collection(P&& p)
  {
    using N0 = std::integral_constant<std::size_t, 0>;
    if constexpr (values::fixed_value_compares_with<collections::size_of<P>, 0>)
      return std::forward<P>(p);
    else
      return collections::views::concat(std::array{get_pattern(p, N0{})}, std::forward<P>(p));
  }

}

#endif
