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
 * \brief Definition for \ref collection::fixed_pattern_collection.
 */

#ifndef OPENKALMAN_COORDINATE_FIXED_PATTERN_COLLECTION_HPP
#define OPENKALMAN_COORDINATE_FIXED_PATTERN_COLLECTION_HPP

#include "collections/collections.hpp"
#include "pattern_collection.hpp"
#include "fixed_pattern.hpp"
#include "fixed_pattern_tuple.hpp"


namespace OpenKalman::coordinates
{
#ifndef __cpp_lib_ranges
  namespace detail
  {
    template<typename T, typename = void>
    struct is_fixed_descriptor_range : std::false_type {};
 
    template<typename T>
    struct is_fixed_descriptor_range<T, std::enable_if_t<
      collections::size_of<T>::value != dynamic_size and
      fixed_pattern<stdcompat::ranges::range_value_t<T>>>>
      : std::true_type {};
  }
#endif 
	
	
  /**
   * \brief An object describing a collection of /ref coordinates::pattern objects.
   * \details This will be a \ref pattern_tuple or a dynamic range over a collection such as std::vector.
   */
  template<typename T>
#ifdef __cpp_lib_ranges
  concept fixed_pattern_collection =
    pattern_collection<T> and
    (fixed_pattern_tuple<T> or
      (collections::size_of_v<T> != dynamic_size and fixed_pattern<stdcompat::ranges::range_value_t<T>>));
#else
  constexpr bool fixed_pattern_collection = collections::collection<T> and
    (fixed_pattern_tuple<T> or detail::is_fixed_descriptor_range<T>::value);
#endif


}

#endif
