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

#ifndef OPENKALMAN_COORDINATES_GROUP_RANGE_HPP
#define OPENKALMAN_COORDINATES_GROUP_RANGE_HPP

#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/ranges.hpp"
#endif
#include "collections/concepts/sized_random_access_range.hpp"
#include "descriptor.hpp"

namespace OpenKalman::coordinate
{
#ifndef __cpp_lib_ranges
  namespace detail
  {
    template<typename T, typename = void>
    struct is_descriptor_range : std::false_type {};
 
    template<typename T>
    struct is_descriptor_range<T, std::enable_if_t<descriptor<ranges::range_value_t<T>>>> : std::true_type {};
  } // namespace detail
#endif 


  /**
   * \brief An object describing a range of \ref collections::descriptor objects.
   * \details This can be, for example, a std::vector containing \ref collections::descriptor "groups".
   */
  template<typename T>
#ifdef __cpp_lib_ranges
  concept descriptor_range = collections::sized_random_access_range<T> and descriptor<std::ranges::range_value_t<T>>;
#else
  constexpr bool descriptor_range = collections::sized_random_access_range<T> and
    detail::is_descriptor_range<std::decay_t<T>>::value;
#endif


} // namespace OpenKalman::coordinate

#endif //OPENKALMAN_COORDINATES_GROUP_RANGE_HPP
