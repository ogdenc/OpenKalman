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
 * \internal
 * \brief Definition of \ref collections::internal::tuple_like_to_tuple.
 */

#ifndef OPENKALMAN_TUPLE_LIKE_TO_TUPLE_HPP
#define OPENKALMAN_TUPLE_LIKE_TO_TUPLE_HPP

#include <type_traits>
#include <tuple>
#include "collections/concepts/uniformly_gettable.hpp"

namespace OpenKalman::collections::internal
{
  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct is_stl_tuple_like : std::false_type {};

    template<typename T>
    struct is_stl_tuple_like<T, std::void_t<decltype(std::tuple_cat(std::declval<T>()))>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<typename T>
    concept stl_tuple_like = requires(T t) { std::tuple_cat(t); };
#else
    template<typename T>
    constexpr bool stl_tuple_like = detail::is_stl_tuple_like<std::decay_t<T>>::value;
#endif


    template<typename Arg, std::size_t...Ix>
    constexpr auto
    tuple_like_to_tuple_impl(Arg&& arg, std::index_sequence<Ix...>)
    {
      return std::tuple {collections::get<Ix>(std::forward<Arg>(arg))...};
    }
  }


  /**
   * \deprecated
   * \brief Convert a \ref uniformly_gettable object to a std::tuple or equivalent
   * \details This is a temporary measure until the expositional "tuple-like" concept in the stl library
   * encompasses \ref tuple_like as defined in this library.
   */
#ifdef __cpp_concepts
  template<uniformly_gettable Arg>
  constexpr detail::stl_tuple_like decltype(auto)
#else
  template<typename Arg, std::enable_if_t<uniformly_gettable<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  tuple_like_to_tuple(Arg&& arg)
  {
    if constexpr (detail::stl_tuple_like<Arg>)
      return std::forward<Arg>(arg);
    else
      return detail::tuple_like_to_tuple_impl(std::forward<Arg>(arg), std::make_index_sequence<size_of_v<Arg>>{});
  }

}

#endif
