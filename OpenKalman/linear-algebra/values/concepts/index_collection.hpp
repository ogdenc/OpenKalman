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
 * \brief Definition for \ref value::index_collection.
 */

#ifndef OPENKALMAN_INDEX_COLLECTION_HPP
#define OPENKALMAN_INDEX_COLLECTION_HPP

#ifdef __cpp_lib_ranges
#include <ranges>
#endif
#include <type_traits>
#include "basics/internal/collection.hpp"
#include "linear-algebra/values/concepts/index_tuple.hpp"

namespace OpenKalman::value
{
#if not defined(__cpp_lib_ranges) or not defined(__cpp_lib_remove_cvref)
  namespace detail
  {
    template<typename T, typename = void>
    struct is_index_range_std : std::false_type {};

    template<typename T>
    struct is_index_range_std<T, std::enable_if_t<value::index<decltype(*std::begin(std::declval<T>()))>>> : std::true_type {};


    template<typename T, typename = void>
    struct is_index_range : std::false_type {};

    template<typename T>
    struct is_index_range<T, std::enable_if_t<value::index<decltype(*begin(std::declval<T>()))>>> : std::true_type {};
  } // namespace detail
#endif


  /**
   * \brief An object describing a collection of /ref value::index objects.
   * \details This will be a tuple-like object or a dynamic range over a collection such as std::vector.
   */
  template<typename T>
#if defined(__cpp_lib_ranges) and defined(__cpp_lib_remove_cvref)
  concept index_collection = OpenKalman::internal::collection<T> and
    (value::index_tuple<T> or value::index<std::ranges::range_value_t<std::remove_cvref_t<T>>>);
#else
  constexpr bool index_collection = OpenKalman::internal::collection<T> and
    (static_vector_space_descriptor_tuple<T> or detail::is_index_range_std<std::decay_t<T>>::value or
      detail::is_index_range<std::decay_t<T>>::value);
#endif

} // namespace OpenKalman::value

#endif //OPENKALMAN_INDEX_COLLECTION_HPP
