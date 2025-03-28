/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition of \ref value::Fixed.
 */

#ifndef OPENKALMAN_VALUE_CLASSES_FIXED_HPP
#define OPENKALMAN_VALUE_CLASSES_FIXED_HPP

#include "values/concepts/value.hpp"
#include "values/functions/to_number.hpp"

namespace OpenKalman::value
{
  /**
   * \internal
   * \brief A defined \ref value::fixed
   * \tparam C A scalar type
   * \tparam constant Optional compile-time arguments for constructing C
   */
#ifdef __cpp_concepts
  template<value::value C, auto...constant> requires std::bool_constant<(C{constant...}, true)>::value
#else
  template<typename C, auto...constant>
#endif
  struct Fixed
  {
    static constexpr auto value {value::to_number(C {constant...})};


    using value_type = std::decay_t<decltype(value)>;


    using type = Fixed;


    constexpr operator value_type() const { return value; }


    constexpr value_type operator()() const { return value; }


    constexpr Fixed() = default;


#ifdef __cpp_concepts
    template<value::fixed T> requires (value::to_number(C {constant...}) == value)
#else
    template<typename T, std::enable_if_t<(value::to_number(C {constant...}) == value), int> = 0>
#endif
    explicit constexpr Fixed(const T&) {};


#ifdef __cpp_concepts
    template<value::fixed T> requires (value::to_number(C {constant...}) == value)
#else
    template<typename T, std::enable_if_t<(value::to_number(C {constant...}) == value), int> = 0>
#endif
    constexpr Fixed& operator=(const T&) { return *this; }

  };


  /**
   * \internal
   * \brief Deduction guide for \ref Fixed where T is already \ref value::fixed.
   */
#ifdef __cpp_concepts
  template<value::fixed T>
#else
  template<typename T, std::enable_if_t<value::fixed<T>, int> = 0>
#endif
  explicit Fixed(const T&) -> Fixed<std::decay_t<T>>;


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_CLASSES_FIXED_HPP
