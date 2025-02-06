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
 * \brief Definition for \ref vector_space_descriptor_range.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_RANGE_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_RANGE_HPP

#ifdef __cpp_lib_ranges
#include <ranges>
#endif
#include "basics/internal/sized_random_access_range.hpp"
#include "vector_space_descriptor.hpp"

namespace OpenKalman::descriptor
{
#ifndef __cpp_lib_ranges
  namespace detail
  {
    template<typename T, typename = void>
    struct is_descriptor_range_std : std::false_type {};
 
    template<typename T>
    struct is_descriptor_range_std<T, std::enable_if_t<vector_space_descriptor<decltype(*std::begin(std::declval<T>()))>>>
      : std::true_type {};


    template<typename T, typename = void>
    struct is_descriptor_range : std::false_type {};

    template<typename T>
    struct is_descriptor_range<T, std::enable_if_t<vector_space_descriptor<decltype(*begin(std::declval<T>()))>>>
      : std::true_type {};
  } // namespace detail
#endif 


  /**
   * \brief An object describing a collection of /ref vector_space_descriptor objects.
   * \details This will be a \ref vector_space_descriptor_tuple or a dynamic range over a collection such as std::vector.
   */
  template<typename T>
#if defined(__cpp_lib_ranges) and defined(__cpp_lib_remove_cvref)
  concept vector_space_descriptor_range = OpenKalman::internal::sized_random_access_range<T> and
    vector_space_descriptor<std::ranges::range_value_t<std::remove_cvref_t<T>>>;
#else
  constexpr bool vector_space_descriptor_range = OpenKalman::internal::sized_random_access_range<T> and
    (detail::is_descriptor_range_std<std::decay_t<T>>::value or detail::is_descriptor_range<std::decay_t<T>>::value);
#endif


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_RANGE_HPP
