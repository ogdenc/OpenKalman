/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and b recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref coordinates::compare.
 */

#ifndef OPENKALMAN_COORDINATES_COMPARE_HPP
#define OPENKALMAN_COORDINATES_COMPARE_HPP

#include "collections/collections.hpp"
#include "coordinates/functions/compare_three_way.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief Compare two \ref coordinates::pattern objects lexicographically.
   * \detail Consecutive \ref euclidean_pattern arguments are consolidated before the comparison occurs.
   * \tparam comp A callable object taking the comparison result (e.g., std::partial_ordering) and returning a bool value
   */
#ifdef __cpp_concepts
  template<pattern A, pattern B, typename Comparison = bool (&)(std::partial_ordering) noexcept> requires
    std::is_invocable_r_v<bool, const Comparison&, stdcompat::partial_ordering>
  constexpr OpenKalman::internal::boolean_testable auto
#else
  template<typename A, typename B, typename Comparison = bool (&)(stdcompat::partial_ordering) noexcept, std::enable_if_t<
    pattern<A> and pattern<B> and
    std::is_invocable_r_v<bool, const Comparison&, stdcompat::partial_ordering>, int> = 0>
  constexpr auto
#endif
  compare(const A& a, const B& b, const Comparison& c = stdcompat::is_eq)
  {
    if constexpr ((not descriptor<A> and not collections::sized<A>) or (not descriptor<B> and not collections::sized<B>))
      return std::false_type {};
    else
      return values::operation(c, compare_three_way(a, b));
  }

}

#endif
