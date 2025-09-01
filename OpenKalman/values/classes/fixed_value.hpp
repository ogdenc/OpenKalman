/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition of \ref values::fixed_value.
 */

#ifndef OPENKALMAN_VALUES_FIXED_VALUE_HPP
#define OPENKALMAN_VALUES_FIXED_VALUE_HPP

#include "values/interface/number_traits.hpp"
#include "values/concepts/complex.hpp"
#include "values/math/real.hpp"
#include "values/math/imag.hpp"

namespace OpenKalman::values
{
  /**
   * \internal
   * \brief A defined \ref values::fixed
   * \tparam C A \ref number, quantity or property
   * \tparam constant Optional compile-time arguments for constructing C
   */
#ifdef __cpp_concepts
  template<typename C, auto...constant> requires
    (sizeof...(constant) == 1 and requires { static_cast<C>((constant,...)); }) or
    (not complex<C> or std::bool_constant<(interface::number_traits<std::decay_t<C>>::make_complex(constant...), sizeof...(constant) == 2)>::value) or
    (complex<C> or std::bool_constant<(C{constant...}, true)>::value)
#else
  template<typename C, auto...constant>
#endif
  struct fixed_value
  {
    static constexpr auto value = []
    {
      if constexpr (sizeof...(constant) == 1) return static_cast<C>((constant,...));
      else if constexpr (complex<C>) return interface::number_traits<std::decay_t<C>>::make_complex(constant...);
      else return C {constant...};
    }();


    using value_type = std::decay_t<decltype(value)>;


    using type = fixed_value;


    constexpr operator value_type() const { return value; }


    constexpr value_type operator()() const { return value; }


    constexpr fixed_value() = default;


#ifdef __cpp_concepts
    template<fixed T> requires (fixed_value_of_v<T> == value)
#else
    template<typename T, std::enable_if_t<fixed_value_of<T>::value == value, int> = 0>
#endif
    explicit constexpr fixed_value(const T&) {};


#ifdef __cpp_concepts
    template<fixed T> requires (fixed_value_of_v<T> == value)
#else
    template<typename T, std::enable_if_t<fixed_value_of<T>::value == value, int> = 0>
#endif
    constexpr fixed_value& operator=(const T&) { return *this; }

  };


  /**
   * \internal
   * \brief Deduction guide for \ref fixed_value where T is already \ref values::fixed but is not \ref values::complex.
   */
#ifdef __cpp_concepts
  template<fixed T> requires (not complex<T>)
#else
  template<typename T, std::enable_if_t<fixed<T> and not complex<T>, int> = 0>
#endif
  explicit fixed_value(const T&) -> fixed_value<typename fixed_value_of<T>::value_type, fixed_value_of_v<T>>;


  /**
   * \internal
   * \brief Deduction guide for \ref fixed_value where T is already \ref values::fixed and is also \ref values::complex.
   */
#ifdef __cpp_concepts
  template<fixed T> requires complex<T>
#else
  template<typename T, std::enable_if_t<fixed<T> and complex<T>, int> = 0>
#endif
  explicit fixed_value(const T&) -> fixed_value<typename fixed_value_of<T>::value_type, values::real(fixed_value_of_v<T>), values::imag(fixed_value_of_v<T>)>;


}


#endif
