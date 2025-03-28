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
#include "basics/language-features.hpp"

namespace OpenKalman::collections
{
#if not defined(__cpp_concepts) or __cpp_generic_lambdas < 201707L or not defined(__cpp_explicit_this_parameter)
  namespace detail
  {
    template<typename T, typename = std::make_index_sequence<std::tuple_size<T>::value>, typename = void>
    struct is_tuple_like_impl : std::false_type {};

    template<typename T, std::size_t...i>
    struct is_tuple_like_impl<T, std::index_sequence<i...>,
      std::void_t<typename std::tuple_element<i, T>::type...>>
      : std::true_type {};


    template<typename T, typename = void>
    struct is_tuple_like : std::false_type {};

    template<typename T>
    struct is_tuple_like<T, std::void_t<decltype(std::tuple_size<T>::value)>> : is_tuple_like_impl<T> {};

  } // namespace detail
#endif


  /**
   * \internal
   * \brief T is a non-empty tuple, pair, array, or other type that can be an argument to std::apply.
   */
  template<typename T>
#if defined(__cpp_concepts) and defined(__cpp_lib_remove_cvref) and __cpp_generic_lambdas >= 201707L
  concept tuple_like = requires(std::remove_cvref_t<T>& t)
  {
    std::tuple_size<std::decay_t<T>>::value;
    requires []<std::size_t...i>(std::index_sequence<i...>)
    { return (... and (
        requires { typename std::tuple_element<i, std::remove_cvref_t<T>>::type; } and
        (requires { t.template get<i>(); } or requires { get<i>(t); } or requires { std::get<i>(t); })));
    } (std::make_index_sequence<std::tuple_size<std::remove_cvref_t<T>>::value>{});
  };
#else
  constexpr bool tuple_like = detail::is_tuple_like<remove_cvref_t<T>>::value;
#endif


} // namespace OpenKalman::collections

#endif //OPENKALMAN_COLLECTIONS_TUPLE_LIKE_HPP
