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

#include "collection.hpp"
#include "uniformly_gettable.hpp"
#include "collections/traits/size_of.hpp"
#include "collections/traits/collection_element.hpp"

namespace OpenKalman::collections
{
  namespace detail
  {
#if not defined(__cpp_concepts) or __cpp_generic_lambdas < 201707L
    template<typename R, typename F, typename = void, typename...Args>
    struct is_invocable_on_range : std::false_type {};

    template<typename R, typename F, typename...Args>
    struct is_invocable_on_range<R, F, std::enable_if_t<stdex::ranges::range<R>>, Args...>
      : std::bool_constant<std::is_invocable_v<F, stdex::ranges::range_value_t<R>, Args...>> {};


    template<typename Tup, typename = void>
    struct is_invocable_on_tuple_sequence { using type = std::index_sequence<>; };

    template<typename Tup>
    struct is_invocable_on_tuple_sequence<Tup, std::enable_if_t<uniformly_gettable<Tup>>>
    { using type = std::make_index_sequence<size_of_v<Tup>>; };


    template<typename Tup, typename F, typename Seq, typename...Args>
    struct is_invocable_on_tuple : std::false_type {};

    template<typename Tup, typename F, std::size_t...Ix, typename...Args>
    struct is_invocable_on_tuple<Tup, F, std::index_sequence<Ix...>, Args...>
      : std::bool_constant<(... and std::is_invocable_v<F&, collection_element_t<Ix, Tup>, Args...>)> {};
#endif

  }


  /**
   * \brief Callable object F is invocable on each element of \ref collection C, with additional parameters Args.
   */
  template<typename F, typename C, typename...Args>
#if defined(__cpp_concepts) and __cpp_generic_lambdas >= 201707L
  concept invocable_on_collection = collection<C> and
    (not stdex::ranges::range<C> or std::regular_invocable<F, stdex::ranges::range_value_t<C>, Args&&...>) and
    (not uniformly_gettable<C> or
      []<std::size_t...i>(std::index_sequence<i...>) {
        return (... and std::regular_invocable<F&, typename collection_element<i, C>::type, Args&&...>);
      } (std::make_index_sequence<size_of_v<C>>{}));
#else
  constexpr bool invocable_on_collection =
    (not stdex::ranges::range<C> or detail::is_invocable_on_range<C, F, void, Args&&...>::value) and
    (not uniformly_gettable<C> or
      detail::is_invocable_on_tuple<C, F, typename detail::is_invocable_on_tuple_sequence<C>::type, Args&&...>::value);
#endif


}

#endif
