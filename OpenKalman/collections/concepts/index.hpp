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
 * \brief Definition for \ref collections::index.
 */

#ifndef OPENKALMAN_COLLECTIONS_INDEX_HPP
#define OPENKALMAN_COLLECTIONS_INDEX_HPP

#include <type_traits>
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/ranges.hpp"
#endif
#include "../../basics/compatibility/language-features.hpp"
#include "values/values.hpp"
#include "collection.hpp"

namespace OpenKalman::collections
{
#if not defined(__cpp_lib_ranges) or not defined(__cpp_lib_remove_cvref) or __cpp_generic_lambdas < 201707L
  namespace detail_index
  {
    template<typename T, std::size_t...Ix>
    constexpr bool is_index_tuple_impl(std::index_sequence<Ix...>) { return (... and values::index<typename std::tuple_element<Ix, T>::type>); }

    template<typename T, typename = void>
    struct is_index_tuple : std::false_type {};

    template<typename T>
    struct is_index_tuple<T, std::enable_if_t<tuple_like<T>>>
      : std::bool_constant<is_index_tuple_impl<T>(std::make_index_sequence<std::tuple_size_v<T>>{})> {};


#ifdef __cpp_lib_remove_cvref
    using std::remove_cvref_t;
#endif
#ifdef __cpp_lib_ranges
    namespace ranges = std::ranges;
#endif


    template<typename T, typename = void>
    struct is_index_range : std::false_type {};

    template<typename T>
    struct is_index_range<T, std::enable_if_t<values::index<ranges::range_value_t<remove_cvref_t<T>>>>> : std::true_type {};

  }
#endif


  /**
   * \brief An object describing a collection of /ref values::index objects.
   * \details This will be a tuple-like object or a dynamic range over a collection such as std::vector.
   */
  template<typename T>
#if defined(__cpp_lib_ranges) and defined(__cpp_lib_remove_cvref) and __cpp_generic_lambdas >= 201707L
  concept index = collection<T> and
    ([]<std::size_t...Ix>(std::index_sequence<Ix...>)
      { return (... and values::index<std::tuple_element_t<Ix, std::remove_cvref_t<T>>>); }
        (std::make_index_sequence<std::tuple_size<std::remove_cvref_t<T>>::value>{}) or
    values::index<std::ranges::range_value_t<std::remove_cvref_t<T>>>);
#else
  constexpr bool index = collection<T> and
    (detail_index::is_index_tuple<std::decay_t<T>>::value or detail_index::is_index_range<std::decay_t<T>>::value);
#endif

} // namespace OpenKalman::collections

#endif //OPENKALMAN_COLLECTIONS_INDEX_HPP
