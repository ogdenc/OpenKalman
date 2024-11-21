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
 * \brief Definition for \value::static_scalar.
 */

#ifndef OPENKALMAN_VALUE_STATIC_SCALAR_CONSTANT_HPP
#define OPENKALMAN_VALUE_STATIC_SCALAR_CONSTANT_HPP

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
    struct has_value_member<T, std::enable_if_t<value::number<decltype(T::value)>>> : std::true_type {};

    namespace detail
    {
      template<typename T, typename = void>
      struct call_result_is_scalar_impl : std::false_type {};

      template<typename T>
      struct call_result_is_scalar_impl<T, std::void_t<std::bool_constant<(T{}(), true)>>>
        : std::bool_constant<value::number<decltype(T{}())>> {};
    } // namespace detail

    template<typename T, typename = void>
    struct call_result_is_scalar : std::false_type {};

    template<typename T>
    struct call_result_is_scalar<T, std::enable_if_t<std::is_default_constructible_v<T>>>
      : std::bool_constant<detail::call_result_is_scalar_impl<T>::value> {};

  } // namespace internal
#endif


  /**
   * \brief T is a scalar constant
   */
  template<typename T>
#ifdef __cpp_concepts
  concept static_scalar = std::default_initializable<std::decay_t<T>> and
    (requires { {std::decay_t<T>::value} -> value::number; } or
      requires {
        {std::decay_t<T>{}()} -> value::number;
        typename std::bool_constant<(std::decay_t<T>{}(), true)>;
      });
#else
  constexpr bool static_scalar = std::is_default_constructible_v<std::decay_t<T>> and
    (internal::has_value_member<std::decay_t<T>>::value or internal::call_result_is_scalar<std::decay_t<T>>::value);
#endif


} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUE_STATIC_SCALAR_CONSTANT_HPP
