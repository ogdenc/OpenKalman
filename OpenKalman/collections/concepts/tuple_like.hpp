/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2021-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref collections::tuple_like.
 */

#ifndef OPENKALMAN_COLLECTIONS_TUPLE_LIKE_HPP
#define OPENKALMAN_COLLECTIONS_TUPLE_LIKE_HPP

#include <tuple>
#include "basics/basics.hpp"
#include "uniformly_gettable.hpp"

namespace OpenKalman::collections
{
#if not defined(__cpp_concepts) and __cpp_generic_lambdas < 201707L
  namespace detail
  {
    template<std::size_t i, typename T, typename = void>
    struct has_tuple_element_impl : std::false_type {};

    template<std::size_t i, typename T>
    struct has_tuple_element_impl<i, T, std::void_t<typename std::tuple_element<i, T>::type>> : std::true_type {};


    template<typename T, typename = std::make_index_sequence<size_of_v<T>>>
    struct is_tuple_like_impl : std::false_type {};

    template<typename T, std::size_t...i>
    struct is_tuple_like_impl<T, std::index_sequence<i...>>
      : std::bool_constant<(... and (gettable<i, T> and has_tuple_element_impl<i, T>::value))> {};


    template<typename T, typename = void>
    struct is_tuple_like : std::false_type {};

    template<typename T>
    struct is_tuple_like<T, std::void_t<decltype(std::tuple_size<T>::value)>> : is_tuple_like_impl<T> {};
  }
#endif


  /**
   * \brief T is a non-empty tuple, pair, array, or other type that acts like a tuple.
   * \details T has defined specializations for std::tuple_size and std::tuple_element, and
   * the elements of T can be accessible by std::get(...), a get(...) member function, or an atd-findable get(...) function.
   */
  template<typename T>
#if defined(__cpp_concepts) and __cpp_generic_lambdas >= 201707L
  concept tuple_like = requires {
      std::tuple_size<std::decay_t<T>>::value;
      requires []<std::size_t...i>(std::index_sequence<i...>) {
        return (... and (gettable<i, T> and requires { typename std::tuple_element<i, std::decay_t<T>>::type; }));
      } (std::make_index_sequence<size_of_v<T>>{});
    };
#else
  constexpr bool tuple_like = detail::is_tuple_like<std::decay_t<T>>::value;
#endif


}

#endif
