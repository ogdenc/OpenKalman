/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref is_square_shaped function.
 */

#ifndef OPENKALMAN_IS_SQUARE_SHAPED_HPP
#define OPENKALMAN_IS_SQUARE_SHAPED_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/traits/get_pattern_collection.hpp"

namespace OpenKalman
{
  /**
   * \brief At least 2 and at most N indices have the same extent.
   * \details N must be at least 2 or must be values::unbounded_size.
   * If the latter, at least two indicess will be compared.
   * If N is greater than the index count, the extents of T will effectively be padded with extent 1
   * so that there are N extents.
   * \note A 0-by-0 array is considered to be square, but a 0-by-1 or 1-by-0 array is not.
   * \return a \ref std::optional which includes the common extent if T is square.
   * \sa square_shaped, patterns::get_common_pattern_collection_dimension
   */
#ifdef __cpp_concepts
  template<auto N = values::unbounded_size, indexible T> requires
    (values::integral<decltype(N)> or stdex::same_as<std::decay_t<decltype(N)>, values::unbounded_size_t>) and
    (not values::integral<decltype(N)> or N >= 2)
#else
  template<std::size_t N = values::unbounded_size, typename T, std::enable_if_t<
    (N == values::unbounded_size or N >= 2) and indexible<T>, int> = 0>
#endif
  constexpr auto
  is_square_shaped(const T& t)
  {
#ifdef __cpp_concepts
    if constexpr (std::same_as<std::decay_t<decltype(N)>, values::unbounded_size_t>)
#else
    if constexpr (N == std::size_t(values::unbounded_size))
#endif
      return patterns::get_common_pattern_collection_dimension<std::max(2_uz, index_count_v<T>)>(
        get_pattern_collection(t));
    else
      return patterns::get_common_pattern_collection_dimension<N>(
        get_pattern_collection(t));


  }


}

#endif
