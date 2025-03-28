/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref coordinate::euclidean_pattern_collection.
 */

#ifndef OPENKALMAN_EUCLIDEAN_PATTERN_COLLECTION_HPP
#define OPENKALMAN_EUCLIDEAN_PATTERN_COLLECTION_HPP

#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/ranges.hpp"
#endif
#include "collections/concepts/collection.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern_tuple.hpp"


namespace OpenKalman::coordinate
{
  /**
   * \brief An object describing a collection of /ref coordinate::pattern objects.
   * \details This will be a \ref pattern_tuple or a dynamic range over a collection such as std::vector.
   */
  template<typename T>
#if defined(__cpp_lib_ranges)
  concept euclidean_pattern_collection = coordinate::pattern_collection<T> and
    (euclidean_pattern_tuple<T> or euclidean_pattern<std::ranges::range_value_t<std::decay_t<T>>>);
#else
  constexpr bool euclidean_pattern_collection = collections::collection<T> and
    (euclidean_pattern_tuple<T> or euclidean_pattern<ranges::range_value_t<std::decay_t<T>>>);
#endif


} // namespace OpenKalman::coordinate

#endif //OPENKALMAN_EUCLIDEAN_PATTERN_COLLECTION_HPP
