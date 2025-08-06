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
 * \brief Definitions relating to the availability of c++ language features.
 */

#ifndef OPENKALMAN_GENERALIZED_STD_GET_HPP
#define OPENKALMAN_GENERALIZED_STD_GET_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

namespace OpenKalman::internal
{
  namespace std_get_detail
  {
    using std::get;


#ifndef __cpp_concepts
    template<std::size_t i, typename T, typename = void>
    struct member_get_is_defined : std::false_type {};

    template<std::size_t i, typename T>
    struct member_get_is_defined<i, T, std::void_t<decltype(std::declval<T>().template get<i>())>> : std::true_type {};


    template<std::size_t i, typename T, typename = void>
    struct function_get_is_defined : std::false_type {};

    template<std::size_t i, typename T>
    struct function_get_is_defined<i, T, std::void_t<decltype(get<i>(std::declval<T>()))>> : std::true_type {};
#endif


    template<std::size_t i>
    struct get_impl
    {
#ifdef __cpp_concepts
      template<typename T> requires
        requires { std::declval<T&&>().template get<i>(); } or
        requires { get<i>(std::declval<T&&>()); }
#else
      template<typename T, std::enable_if_t<member_get_is_defined<i, T>::value or function_get_is_defined<i, T>::value, int> = 0>
#endif
      constexpr decltype(auto)
      operator() [[nodiscard]] (T&& t) const
      {
#ifdef __cpp_concepts
        if constexpr (requires { std::forward<T>(t).template get<i>(); })
#else
        if constexpr (member_get_is_defined<i, T>::value)
#endif
          return std::forward<T>(t).template get<i>();
        else
          return get<i>(std::forward<T>(t));
      }
    };
  }


  /**
   * \internal
   * \brief This is a placeholder for a more general <code>std::get</code> function that might be added to the standard library, possibly by another name.
   */
  template<std::size_t i>
  inline constexpr std_get_detail::get_impl<i>
  generalized_std_get;

}


#endif