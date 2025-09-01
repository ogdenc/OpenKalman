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
 * \brief Definition for \ref coordinates::euclidean_pattern_collection.
 */

#ifndef OPENKALMAN_EUCLIDEAN_PATTERN_COLLECTION_HPP
#define OPENKALMAN_EUCLIDEAN_PATTERN_COLLECTION_HPP

#include "collections/collections.hpp"
#include "coordinates/concepts/euclidean_pattern.hpp"
#include "coordinates/concepts/euclidean_pattern_tuple.hpp"

namespace OpenKalman::coordinates
{
#ifndef __cpp_lib_ranges
  namespace detail
  {
    template<typename T, typename = void>
    struct is_euclidean_descriptor_range : std::false_type {};

    template<typename T>
    struct is_euclidean_descriptor_range<T, std::enable_if_t<euclidean_pattern<stdcompat::ranges::range_value_t<T>>>>
      : std::true_type {};
  }
#endif


/**
   * \brief An object describing a collection of /ref coordinates::pattern objects.
   * \details This will be a \ref pattern_tuple or a dynamic range over a collection such as std::vector.
   */
  template<typename T>
#if defined(__cpp_lib_ranges)
  concept euclidean_pattern_collection = coordinates::pattern_collection<T> and
    (euclidean_pattern_tuple<T> or euclidean_pattern<stdcompat::ranges::range_value_t<T>>);
#else
  constexpr bool euclidean_pattern_collection = collections::collection<T> and
    (euclidean_pattern_tuple<T> or detail::is_euclidean_descriptor_range<T>::value);
#endif


}

#endif
