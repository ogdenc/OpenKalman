/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and b recursive filters.
 *
 * Copyright (c) 2025-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref patterns::compare.
 */

#ifndef OPENKALMAN_PATTERNS_COMPARE_HPP
#define OPENKALMAN_PATTERNS_COMPARE_HPP

#include "collections/collections.hpp"
#include "patterns/concepts/pattern.hpp"
#include "patterns/functions/compare_three_way.hpp"

namespace OpenKalman::patterns
{
  /**
   * \brief Compare two \ref patterns::pattern objects lexicographically.
   * \detail Consecutive \ref euclidean_pattern arguments are consolidated before the comparison occurs.
   * \tparam comp A callable object taking the comparison result (e.g., std::partial_ordering) and returning a bool value
   */
#ifdef __cpp_concepts
  template<auto comp = &stdex::is_eq, pattern A, pattern B> requires
    std::is_invocable_r_v<bool, decltype(comp), stdex::partial_ordering>
  constexpr OpenKalman::internal::boolean_testable auto
#else
  template<auto comp = &stdex::is_eq, typename A, typename B, std::enable_if_t<
    pattern<A> and pattern<B> and
    std::is_invocable_r_v<bool, decltype(comp), stdex::partial_ordering>, int> = 0>
  constexpr auto
#endif
  compare(const A& a, const B& b)
  {
    return values::operation(comp, compare_three_way(a, b));
  }

}

#endif
