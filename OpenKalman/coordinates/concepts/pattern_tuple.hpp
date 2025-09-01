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
 * \brief Definition for \ref pattern_tuple.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_TUPLE_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_TUPLE_HPP

#include <type_traits>
#include <tuple>
#include <utility>
#include "collections/collections.hpp"
#include "pattern.hpp"

namespace OpenKalman::coordinates
{
#if not defined(__cpp_concepts) or __cpp_generic_lambdas < 201707L
  namespace detail
  {
    template<typename T, std::size_t...Ix>
    constexpr bool is_pattern_tuple_impl(std::index_sequence<Ix...>)
    {
      return (... and pattern<collections::collection_element_t<Ix, T>>);
    }

    template<typename T, typename = void>
    struct is_pattern_tuple : std::false_type {};

    template<typename T>
    struct is_pattern_tuple<T, std::enable_if_t<collections::uniformly_gettable<T>>>
      : std::bool_constant<is_pattern_tuple_impl<T>(std::make_index_sequence<collections::size_of_v<T>>{})> {};
  }
#endif

	
  /**
   * \brief An object describing a tuple-like collection of /ref coordinates::pattern objects.
   */
  template<typename T>
#if defined(__cpp_concepts) and __cpp_generic_lambdas >= 201707L
  concept pattern_tuple = collections::uniformly_gettable<T> and
    []<std::size_t...Ix>(std::index_sequence<Ix...>)
      { return (... and pattern<collections::collection_element_t<Ix, std::decay_t<T>>>); }
      (std::make_index_sequence<collections::size_of_v<T>>{});
#else
  constexpr bool pattern_tuple = detail::is_pattern_tuple<std::decay_t<T>>::value;
#endif


}

#endif
