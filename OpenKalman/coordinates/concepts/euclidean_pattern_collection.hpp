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
 * \brief Definition for \ref coordinates::euclidean_pattern_collection.
 */

#ifndef OPENKALMAN_EUCLIDEAN_PATTERN_COLLECTION_HPP
#define OPENKALMAN_EUCLIDEAN_PATTERN_COLLECTION_HPP

#include "collections/collections.hpp"
#include "euclidean_pattern.hpp"

namespace OpenKalman::coordinates
{
#if not defined(__cpp_concepts) or __cpp_generic_lambdas < 201707L
  namespace detail
  {
    template<typename T, std::size_t...Ix>
    constexpr bool is_euclidean_pattern_iter_impl(std::index_sequence<Ix...>)
    {
      return (... and euclidean_pattern<collections::collection_element_t<Ix, T>>);
    }


    template<typename T, typename = void>
    struct is_euclidean_pattern_iter : std::false_type {};

    template<typename T>
    struct is_euclidean_pattern_iter<T, std::enable_if_t<collections::uniformly_gettable<T>>>
      : std::bool_constant<is_euclidean_pattern_iter_impl<T>(std::make_index_sequence<collections::size_of_v<T>>{})> {};


    template<typename T, typename = void>
    struct is_euclidean_descriptor_range : std::false_type {};

    template<typename T>
    struct is_euclidean_descriptor_range<T, std::enable_if_t<euclidean_pattern<stdcompat::ranges::range_value_t<T>>>>
      : std::true_type {};
  }
#endif


/**
   * \brief An object describing a collection of /ref euclidean_pattern objects.
   */
  template<typename T>
#if defined(__cpp_concepts) and __cpp_generic_lambdas >= 201707L
  concept euclidean_pattern_collection =
    collections::collection<T> and
    ( euclidean_pattern<stdcompat::ranges::range_value_t<T>> or
      []<std::size_t...Ix>(std::index_sequence<Ix...>)
        { return (... and euclidean_pattern<collections::collection_element_t<Ix, T>>); }
        (std::make_index_sequence<collections::size_of_v<T>>{})
    );
#else
  constexpr bool euclidean_pattern_collection =
    collections::collection<T> and
    ( detail::is_euclidean_pattern_iter<T>::value or
      detail::is_euclidean_descriptor_range<T>::value);
#endif


}

#endif
