/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \values::fixed.
 */

#ifndef OPENKALMAN_VALUES_FIXED_HPP
#define OPENKALMAN_VALUES_FIXED_HPP

#include "basics/basics.hpp"

namespace OpenKalman::values
{
#ifndef __cpp_concepts
  namespace internal
  {
    // These functions are also used in values::to_value_type

    template<typename T, typename = void>
    struct has_value_member : std::false_type {};

    template<typename T>
    struct has_value_member<T, std::void_t<decltype(T::value)>> : std::true_type {};


    template<typename T, typename = void>
    struct call_result_is_defined : std::false_type {};

    template<typename T>
    struct call_result_is_defined<T, std::void_t<decltype(std::declval<T>()())>> : std::true_type {};


    template<typename T, typename = void>
    struct call_result_is_constexpr : std::false_type {};

    template<typename T>
    struct call_result_is_constexpr<T, std::void_t<std::bool_constant<(T{}(), true)>>> : std::true_type {};

  }
#endif


  /**
   * \brief T is a \ref value that is determinable at compile time.
   * \todo Include objects that can be implicitly converted to a number at compile time?
   */
  template<typename T>
#ifdef __cpp_concepts
  concept fixed =
    std::default_initializable<std::decay_t<T>> and
    (requires { std::decay_t<T>::value; } or
      requires() {
        std::decay_t<T>{}();
        typename std::bool_constant<(std::decay_t<T>{}(), true)>;
      });
#else
  constexpr bool fixed =
    stdcompat::default_initializable<std::decay_t<T>> and
    (internal::has_value_member<std::decay_t<T>>::value or
      (internal::call_result_is_defined<std::decay_t<T>>::value and internal::call_result_is_constexpr<std::decay_t<T>>::value));
#endif


}

#endif
