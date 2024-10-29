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

#include "basics/global-definitions.hpp"
#include "basics/values/scalars/scalar_constant.hpp"
#include "basics/values/internal/scalar_constant_operation.hpp"

namespace OpenKalman::values
{
#ifdef __cpp_concepts
  template<scalar_constant<ConstantType::static_constant> Arg>
#else
  template<typename Arg, std::enable_if_t<scalar_constant<Arg>, int> = 0>
#endif
  constexpr Arg&& operator+(Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<scalar_constant<ConstantType::static_constant> Arg>
#else
  template<typename Arg, std::enable_if_t<scalar_constant<Arg, ConstantType::static_constant>, int> = 0>
#endif
  constexpr auto operator-(const Arg& arg)
  {
    return values::scalar_constant_operation {std::negate<>{}, arg};
  }


#ifdef __cpp_concepts
  template<scalar_constant<ConstantType::static_constant> Arg1, scalar_constant<ConstantType::static_constant> Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<scalar_constant<Arg1, ConstantType::static_constant> and scalar_constant<Arg2, ConstantType::static_constant>, int> = 0>
#endif
  constexpr auto operator+(const Arg1& arg1, const Arg2& arg2)
  {
    return values::scalar_constant_operation {std::plus<>{}, arg1, arg2};
  }


#ifdef __cpp_concepts
  template<scalar_constant<ConstantType::static_constant> Arg1, scalar_constant<ConstantType::static_constant> Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<scalar_constant<Arg1, ConstantType::static_constant> and scalar_constant<Arg2, ConstantType::static_constant>, int> = 0>
#endif
  constexpr auto operator-(const Arg1& arg1, const Arg2& arg2)
  {
    return values::scalar_constant_operation {std::minus<>{}, arg1, arg2};
  }


#ifdef __cpp_concepts
  template<scalar_constant<ConstantType::static_constant> Arg1, scalar_constant<ConstantType::static_constant> Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<scalar_constant<Arg1, ConstantType::static_constant> and scalar_constant<Arg2, ConstantType::static_constant>, int> = 0>
#endif
  constexpr auto operator*(const Arg1& arg1, const Arg2& arg2)
  {
    return values::scalar_constant_operation {std::multiplies<>{}, arg1, arg2};
  }


#ifdef __cpp_concepts
  template<scalar_constant<ConstantType::static_constant> Arg1, scalar_constant<ConstantType::static_constant> Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<scalar_constant<Arg1, ConstantType::static_constant> and scalar_constant<Arg2, ConstantType::static_constant>, int> = 0>
#endif
  constexpr auto operator/(const Arg1& arg1, const Arg2& arg2)
  {
    return values::scalar_constant_operation {std::divides<>{}, arg1, arg2};
  }


} // namespace OpenKalman::values

#endif //OPENKALMAN_SCALAR_ARITHMETIC_HPP
