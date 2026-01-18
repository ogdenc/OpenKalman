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
 * \brief Definition of \ref is_one_dimensional function.
 */

#ifndef OPENKALMAN_IS_ONE_DIMENSIONAL_HPP
#define OPENKALMAN_IS_ONE_DIMENSIONAL_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/traits/get_pattern_collection.hpp"

namespace OpenKalman
{
  /**
   * \brief Determine whether T is one_dimensional, meaning that every index has a dimension of 1.
   * \details N must be a non-negative number or \ref values::unbounded_size.
   * Note that the dimension of any indices greater than \ref index_count_v<T> will naturally be 1.
   */
#ifdef __cpp_concepts
  template<auto N = values::unbounded_size, indexible T> requires
    (values::integral<decltype(N)> or stdex::same_as<std::decay_t<decltype(N)>, values::unbounded_size_t>) and
    (not values::integral<decltype(N)> or N >= 0)
  constexpr internal::boolean_testable auto
#else
  template<std::size_t N = values::unbounded_size, typename T, std::enable_if_t<
    (N == values::unbounded_size or N >= 0) and indexible<T>, int> = 0>
  constexpr auto
#endif
  is_one_dimensional(const T& t)
  {
    return patterns::compare_collection_patterns_with_dimension<1_uz, &stdex::is_eq, N>(get_pattern_collection(t));
  }

}

#endif
