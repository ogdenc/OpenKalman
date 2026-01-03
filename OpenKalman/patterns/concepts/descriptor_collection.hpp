/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref patterns::descriptor_collection.
 */

#ifndef OPENKALMAN_PATTERNS_GROUP_COLLECTION_HPP
#define OPENKALMAN_PATTERNS_GROUP_COLLECTION_HPP

#include "collections/collections.hpp"
#include "descriptor.hpp"

namespace OpenKalman::patterns
{
#if not defined(__cpp_concepts) or __cpp_generic_lambdas < 201707L or not defined(__cpp_lib_ranges)
  namespace detail
  {
    template<typename T, typename = void>
    struct is_descriptor_range : std::false_type {};

    template<typename T>
    struct is_descriptor_range<T, std::enable_if_t<descriptor<stdex::ranges::range_value_t<T>>>> : std::true_type {};


    template<typename T>
    constexpr bool descriptor_range =
      stdex::ranges::random_access_range<T> and is_descriptor_range<T>::value;


    template<typename T, std::size_t...Ix>
    constexpr bool is_descriptor_tuple_impl(std::index_sequence<Ix...>)
    {
      return (... and descriptor<collections::collection_element_t<Ix, T>>);
    }

    template<typename T, typename = void>
    struct is_descriptor_tuple : std::false_type {};

    template<typename T>
    struct is_descriptor_tuple<T, std::enable_if_t<collections::uniformly_gettable<T>>>
      : std::bool_constant<is_descriptor_tuple_impl<T>(std::make_index_sequence<collections::size_of_v<T>>{})> {};


    template<typename T>
    inline constexpr bool descriptor_tuple = is_descriptor_tuple<std::decay_t<T>>::value;
  }
#endif


  /**
   * \brief An object describing a collection of /ref patterns::descriptor objects.
   * \details This can be either a \ref uniformly_gettable structure or a \ref collections::sized "sized" std::ranges::random_access_range.
   */
  template<typename T>
#if defined(__cpp_concepts) and __cpp_generic_lambdas >= 201707L and defined(__cpp_lib_ranges)
  concept descriptor_collection =
    (std::ranges::random_access_range<T> and descriptor<std::ranges::range_value_t<T>>) or
    (collections::uniformly_gettable<T> and
      []<std::size_t...Ix>(std::index_sequence<Ix...>)
        { return (... and descriptor<collections::collection_element_t<Ix, T>>); }
          (std::make_index_sequence<collections::size_of_v<T>>{}));
#else
  constexpr bool descriptor_collection =
    detail::descriptor_range<T> or
    detail::descriptor_tuple<T>;
#endif


}

#endif
