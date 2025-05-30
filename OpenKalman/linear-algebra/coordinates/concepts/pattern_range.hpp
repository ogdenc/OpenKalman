/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref pattern_range.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_RANGE_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_RANGE_HPP

#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/ranges.hpp"
#endif
#include "collections/concepts/sized_random_access_range.hpp"
#include "pattern.hpp"

namespace OpenKalman::coordinates
{
#ifndef __cpp_lib_ranges
  namespace detail
  {
    template<typename T, typename = void>
    struct is_pattern_range : std::false_type {};

    template<typename T>
    struct is_pattern_range<T, std::enable_if_t<pattern<ranges::range_value_t<T>>>>
      : std::true_type {};
  } // namespace detail
#endif 


  /**
   * \brief An object describing a collection of /ref pattern objects.
   * \details This will be a \ref pattern_tuple or a dynamic range over a collection such as std::vector.
   */
  template<typename T>
#if defined(__cpp_lib_ranges)
  concept pattern_range = collections::sized_random_access_range<T> and pattern<std::ranges::range_value_t<T>>;
#else
  constexpr bool pattern_range = collections::sized_random_access_range<T> and detail::is_pattern_range<T>::value;
#endif


} // namespace OpenKalman::coordinates

#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_RANGE_HPP
