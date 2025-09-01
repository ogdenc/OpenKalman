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

#include "values/values.hpp"
#include "collection.hpp"
#include "collections/traits/size_of.hpp"
#include "collections/traits/collection_element.hpp"

namespace OpenKalman::collections
{
#if not defined(__cpp_lib_ranges) or __cpp_generic_lambdas < 201707L
  namespace detail
  {
    template<typename T, typename = void>
    struct is_index_range : std::false_type {};

    template<typename T>
    struct is_index_range<T, std::enable_if_t<values::index<stdcompat::ranges::range_value_t<T>>>> : std::true_type {};


    template<std::size_t i, typename T, typename = void>
    struct has_index_element : std::false_type {};

    template<std::size_t i, typename T>
    struct has_index_element<i, T, std::enable_if_t<values::index<typename collection_element<i, T>::type>>>
      : std::true_type {};


    template<typename T, typename = std::make_index_sequence<size_of_v<T>>>
    struct is_index_tuple_impl : std::false_type {};

    template<typename T, std::size_t...i>
    struct is_index_tuple_impl<T, std::index_sequence<i...>>
      : std::bool_constant<(... and has_index_element<i, T>::value)> {};


    template<typename T, typename = void>
    struct is_index_tuple : std::false_type {};

    template<typename T>
    struct is_index_tuple<T, std::enable_if_t<size_of<T>::value != dynamic_size>> : is_index_tuple_impl<T> {};
  }
#endif


  /**
   * \brief An object describing a collection of /ref values::index objects.
   */
  template<typename T>
#if defined(__cpp_lib_ranges) and __cpp_generic_lambdas >= 201707L
  concept index =
    collection<T> and
    (values::index<std::ranges::range_value_t<T>> or (
      size_of_v<T> != dynamic_size and
      []<std::size_t...Ix>(std::index_sequence<Ix...>) {
          return (... and values::index<typename collection_element<Ix, T>::type>);
        }(std::make_index_sequence<size_of<T>::value>{}))
    );
#else
  constexpr bool index =
    collection<T> and (detail::is_index_range<T>::value or detail::is_index_tuple<T>::value);
#endif

}

#endif
