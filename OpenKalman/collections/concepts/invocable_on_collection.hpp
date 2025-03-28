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
 * \brief Definition for \ref invocable_on_collection.
 */

#ifndef OPENKALMAN_INVOCABLE_ON_COLLECTION_HPP
#define OPENKALMAN_INVOCABLE_ON_COLLECTION_HPP

#include "tuple_like.hpp"
#include "sized_random_access_range.hpp"

namespace OpenKalman::collections
{
  namespace detail
  {
#ifdef __cpp_lib_ranges
    template<typename Tup, typename F, typename = std::make_index_sequence<std::tuple_size_v<std::decay_t<Tup>>>>
    struct is_invocable_on_tuple : std::false_type {};

    template<typename Tup, typename F, std::size_t...Ix>
    struct is_invocable_on_tuple<Tup, F, std::index_sequence<Ix...>>
      : std::bool_constant<(... and std::regular_invocable<F&, std::tuple_element_t<Ix, Tup>>)> {};
#else
    template<typename Tup, typename = void>
    struct is_invocable_on_tuple_sequence { using type = std::index_sequence<>; };

    template<typename Tup>
    struct is_invocable_on_tuple_sequence<Tup, std::enable_if_t<tuple_like<Tup>>>
    { using type = std::make_index_sequence<std::tuple_size_v<std::decay_t<Tup>>>; };


    template<typename Tup, typename F, typename = typename is_invocable_on_tuple_sequence<Tup>::type>
    struct is_invocable_on_tuple : std::false_type {};

    template<typename Tup, typename F, std::size_t...Ix>
    struct is_invocable_on_tuple<Tup, F, std::index_sequence<Ix...>>
      : std::bool_constant<(... and std::is_invocable_v<F&, std::tuple_element_t<Ix, Tup>>)> {};


    template<typename C, typename F, typename = void>
    struct is_invocable_on_range : std::false_type {};

    template<typename C, typename F>
    struct is_invocable_on_range<C, F, std::enable_if_t<ranges::range<C>>>
      : std::bool_constant<std::is_invocable_v<F, ranges::range_value_t<C>>> {};
#endif

  } // namespace detail


  /**
   * \brief Callable object F is invocable on each element of \ref collection C.
   */
  template<typename F, typename C>
#ifdef __cpp_lib_ranges
  concept invocable_on_collection = collection<C> and
    (not std::ranges::range<C> or std::regular_invocable<F, std::ranges::range_value_t<C>>) and
    (not tuple_like<C> or detail::is_invocable_on_tuple<C, F>::value);
#else
  constexpr bool invocable_on_collection =
    (not sized_random_access_range<C> or detail::is_invocable_on_range<C, F>::value) and
    (not tuple_like<C> or detail::is_invocable_on_tuple<C, F>::value);
#endif


} // namespace OpenKalman::collections

#endif //OPENKALMAN_INVOCABLE_ON_COLLECTION_HPP
