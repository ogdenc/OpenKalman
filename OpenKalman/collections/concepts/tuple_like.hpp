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
#if not defined(__cpp_concepts) or __cpp_generic_lambdas < 201707L
  namespace detail
  {
    template<typename T, typename = void>
    struct is_tuple_like : std::false_type {};

    template<typename T>
    struct is_tuple_like<T, std::void_t<decltype(std::tuple_size<T>::value)>> : std::true_type {};

  } // namespace detail
#endif


  /**
   * \brief T is a non-empty tuple, pair, array, or other type that acts like a tuple.
   * \details T has defined specializations for std::tuple_size and std::tuple_element, and
   * the elements of T can be accessible by std::get(...), a get(...) member function, or an atd-findable get(...) function.
   */
  template<typename T>
#if defined(__cpp_concepts) and __cpp_generic_lambdas >= 201707L
  concept tuple_like = uniformly_gettable<T> and requires
  {
    typename std::tuple_size<std::decay_t<T>>;
    std::tuple_size<std::decay_t<T>>::value;
  };
#else
  constexpr bool tuple_like = uniformly_gettable<T> and detail::is_tuple_like<std::decay_t<T>>::value;
#endif


} // namespace OpenKalman::collections

#endif //OPENKALMAN_COLLECTIONS_TUPLE_LIKE_HPP
