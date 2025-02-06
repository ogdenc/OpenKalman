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
 * \internal
 * \brief Definition for \ref internal::tuple_like.
 */

#ifndef OPENKALMAN_TUPLE_LIKE_HPP
#define OPENKALMAN_TUPLE_LIKE_HPP

#include <tuple>

namespace OpenKalman::internal
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_tuple_like : std::false_type {};

    template<typename T>
    struct is_tuple_like<T, std::void_t<typename std::tuple_size<T>::value_type>> : std::true_type {};
  }
#endif


  /**
   * \internal
   * \brief T is a non-empty tuple, pair, array, or other type that can be an argument to std::apply.
   */
#if defined(__cpp_concepts) and defined(__cpp_lib_remove_cvref)
  template<typename T>
  concept tuple_like = requires { typename std::tuple_size<std::remove_cvref_t<T>>::value_type; };
#else
  template<typename T>
  constexpr bool tuple_like = detail::is_tuple_like<std::decay_t<T>>::value;
#endif

} // namespace OpenKalman::internal

#endif //OPENKALMAN_TUPLE_LIKE_HPP
