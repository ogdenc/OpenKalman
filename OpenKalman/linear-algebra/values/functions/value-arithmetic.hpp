/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for value arithmetic.
 */

#ifndef OPENKALMAN_VALUE_ARITHMETIC_HPP
#define OPENKALMAN_VALUE_ARITHMETIC_HPP

#include <functional>
#include "linear-algebra/values/concepts/fixed.hpp"
#include "linear-algebra/values/concepts/value.hpp"
#include "to_number.hpp"
#include "linear-algebra/values/classes/operation.hpp"

namespace OpenKalman::value
{
#ifdef __cpp_concepts
  template<value::value Arg>
#else
  template<typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
#endif
  constexpr Arg&& operator+(Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<value::value Arg>
#else
  template<typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
#endif
  constexpr auto operator-(const Arg& arg)
  {
    if constexpr (value::fixed<Arg>)
      return value::operation {std::negate<>{}, arg};
    else
      return -value::to_number(arg);
  }


#ifdef __cpp_concepts
  template<value::value Arg1, value::value Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<value::value<Arg1> and value::value<Arg2>, int> = 0>
#endif
  constexpr auto operator+(const Arg1& arg1, const Arg2& arg2)
  {
    if constexpr (value::fixed<Arg1> and value::fixed<Arg2>)
      return value::operation {std::plus<>{}, arg1, arg2};
    else
      return value::to_number(arg1) + value::to_number(arg2);
  }


#ifdef __cpp_concepts
  template<value::value Arg1, value::value Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<value::value<Arg1> and value::value<Arg2>, int> = 0>
#endif
  constexpr auto operator-(const Arg1& arg1, const Arg2& arg2)
  {
    if constexpr (value::fixed<Arg1> and value::fixed<Arg2>)
      return value::operation {std::minus<>{}, arg1, arg2};
    else
      return value::to_number(arg1) - value::to_number(arg2);
  }


#ifdef __cpp_concepts
  template<value::value Arg1, value::value Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<value::value<Arg1> and value::value<Arg2>, int> = 0>
#endif
  constexpr auto operator*(const Arg1& arg1, const Arg2& arg2)
  {
    if constexpr (value::fixed<Arg1> and value::fixed<Arg2>)
      return value::operation {std::multiplies<>{}, arg1, arg2};
    else
      return value::to_number(arg1) * value::to_number(arg2);
  }


#ifdef __cpp_concepts
  template<value::value Arg1, value::value Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<value::value<Arg1> and value::value<Arg2>, int> = 0>
#endif
  constexpr auto operator/(const Arg1& arg1, const Arg2& arg2)
  {
    if constexpr (value::fixed<Arg1> and value::fixed<Arg2>)
      return value::operation {std::divides<>{}, arg1, arg2};
    else
      return value::to_number(arg1) / value::to_number(arg2);
  }


#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  template<value::value A, value::value B>
  constexpr auto operator<=>(const A& a, const B& b)
  {
    return value::to_number(a) <=> value::to_number(b);
  }

  template<value::value A, value::value B>
  constexpr bool operator==(const A& a, const B& b)
  {
    return std::is_eq(a <=> b);
  }
#else
  template<typename A, typename B, std::enable_if_t<value::value<A> and value::value<B>, int> = 0>
  constexpr bool operator==(const A& a, const B& b)
  {
    return value::to_number(a) == value::to_number(b);
  }

  template<typename A, typename B, std::enable_if_t<value::value<A> and value::value<B>, int> = 0>
  constexpr bool operator!=(const A& a, const B& b)
  {
    return value::to_number(a) != value::to_number(b);
  }

  template<typename A, typename B, std::enable_if_t<value::value<A> and value::value<B>, int> = 0>
  constexpr auto operator<(const A& a, const B& b)
  {
    return value::to_number(a) < value::to_number(b);
  }

  template<typename A, typename B, std::enable_if_t<value::value<A> and value::value<B>, int> = 0>
  constexpr auto operator>(const A& a, const B& b)
  {
    return value::to_number(a) > value::to_number(b);
  }

  template<typename A, typename B, std::enable_if_t<value::value<A> and value::value<B>, int> = 0>
  constexpr auto operator<=(const A& a, const B& b)
  {
    return value::to_number(a) <= value::to_number(b);
  }

  template<typename A, typename B, std::enable_if_t<value::value<A> and value::value<B>, int> = 0>
  constexpr auto operator>=(const A& a, const B& b)
  {
    return value::to_number(a) >= value::to_number(b);
  }
#endif


} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUE_ARITHMETIC_HPP
