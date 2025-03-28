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
 * \brief Definition for \value::fixed.
 */

#ifndef OPENKALMAN_VALUE_FIXED_HPP
#define OPENKALMAN_VALUE_FIXED_HPP

#ifdef __cpp_concepts
#include <concepts>
#endif
#include <type_traits>
#include "number.hpp"

namespace OpenKalman::value
{
#ifndef __cpp_concepts
  namespace internal
  {
    // These functions are also used in value::to_number

    template<typename T, typename = void>
    struct has_value_member : std::false_type {};

    template<typename T>
    struct has_value_member<T, std::enable_if_t<number<decltype(T::value)>>> : std::true_type {};

    template<typename T, typename = void>
    struct call_result_is_fixed : std::false_type {};

    template<typename T>
    struct call_result_is_fixed<T, std::void_t<std::bool_constant<(T{}(), true)>>> : std::true_type {};

  } // namespace internal
#endif


  /**
   * \brief T is a value::value that is determinable at compile time.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept fixed = std::default_initializable<std::decay_t<T>> and
    (requires { {std::decay_t<T>::value} -> number; } or
      requires {
        {std::decay_t<T>{}()} -> number;
        typename std::bool_constant<(std::decay_t<T>{}(), true)>;
      });
#else
  constexpr bool fixed = std::is_default_constructible_v<std::decay_t<T>> and
    (internal::has_value_member<std::decay_t<T>>::value or internal::call_result_is_fixed<std::decay_t<T>>::value);
#endif


} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUE_FIXED_HPP
