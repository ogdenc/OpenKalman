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
 * \brief Definitions for scalar arithmetic.
 */

#ifndef OPENKALMAN_SCALAR_ARITHMETIC_HPP
#define OPENKALMAN_SCALAR_ARITHMETIC_HPP

#include <functional>
#include "linear-algebra/values/concepts/static_scalar.hpp"
#include "linear-algebra/values/concepts/scalar.hpp"
#include "linear-algebra/values/internal-classes/static_scalar_operation.hpp"

namespace OpenKalman::value
{
#ifdef __cpp_concepts
  template<value::scalar Arg>
#else
  template<typename Arg, std::enable_if_t<value::scalar<Arg>, int> = 0>
#endif
  constexpr Arg&& operator+(Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<value::scalar Arg>
#else
  template<typename Arg, std::enable_if_t<value::scalar<Arg>, int> = 0>
#endif
  constexpr auto operator-(const Arg& arg)
  {
    if constexpr (value::static_scalar<Arg>)
      return value::static_scalar_operation {std::negate<>{}, arg};
    else
      return -value::to_number(arg);
  }


#ifdef __cpp_concepts
  template<value::scalar Arg1, value::scalar Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<value::scalar<Arg1> and value::scalar<Arg2>, int> = 0>
#endif
  constexpr auto operator+(const Arg1& arg1, const Arg2& arg2)
  {
    if constexpr (value::static_scalar<Arg1> and value::static_scalar<Arg2>)
      return value::static_scalar_operation {std::plus<>{}, arg1, arg2};
    else
      return value::to_number(arg1) + value::to_number(arg2);
  }


#ifdef __cpp_concepts
  template<value::scalar Arg1, value::scalar Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<value::scalar<Arg1> and value::scalar<Arg2>, int> = 0>
#endif
  constexpr auto operator-(const Arg1& arg1, const Arg2& arg2)
  {
    if constexpr (value::static_scalar<Arg1> and value::static_scalar<Arg2>)
      return value::static_scalar_operation {std::minus<>{}, arg1, arg2};
    else
      return value::to_number(arg1) - value::to_number(arg2);
  }


#ifdef __cpp_concepts
  template<value::scalar Arg1, value::scalar Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<value::scalar<Arg1> and value::scalar<Arg2>, int> = 0>
#endif
  constexpr auto operator*(const Arg1& arg1, const Arg2& arg2)
  {
    if constexpr (value::static_scalar<Arg1> and value::static_scalar<Arg2>)
      return value::static_scalar_operation {std::multiplies<>{}, arg1, arg2};
    else
      return value::to_number(arg1) * value::to_number(arg2);
  }


#ifdef __cpp_concepts
  template<value::scalar Arg1, value::scalar Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<value::scalar<Arg1> and value::scalar<Arg2>, int> = 0>
#endif
  constexpr auto operator/(const Arg1& arg1, const Arg2& arg2)
  {
    if constexpr (value::static_scalar<Arg1> and value::static_scalar<Arg2>)
      return value::static_scalar_operation {std::divides<>{}, arg1, arg2};
    else
      return value::to_number(arg1) / value::to_number(arg2);
  }


} // namespace OpenKalman::value

#endif //OPENKALMAN_SCALAR_ARITHMETIC_HPP
