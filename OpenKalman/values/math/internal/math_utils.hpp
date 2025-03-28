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
 * \brief Definition for various utilities for math functions.
 */

#ifndef OPENKALMAN_VALUE_MATH_UTILS_HPP
#define OPENKALMAN_VALUE_MATH_UTILS_HPP

#include <type_traits>
#include <limits>
#include "basics/language-features.hpp"
#include "values/math/signbit.hpp"
#include "values/math/copysign.hpp"
#include "values/math/sqrt.hpp"
#include "asin_utils.hpp"

namespace OpenKalman::value::internal
{
  template<typename T>
  constexpr T atan_impl(const T& x)
  {
    return asin_impl(x / value::sqrt(1 + x * x));
  }


  template<typename T>
  constexpr T atan2_impl(const T& y, const T& x)
  {
    constexpr auto pi = numbers::pi_v<T>;

    if constexpr (std::numeric_limits<T>::has_infinity)
    {
      constexpr auto inf = std::numeric_limits<T>::infinity();
      if (y == +inf)
      {
        if (x == +inf) return pi/4;
        else if (x == -inf) return 3*pi/4;
        else return pi/2;
      }
      else if (y == -inf)
      {
        if (x == +inf) return -pi/4;
        else if (x == -inf) return -3*pi/4;
        else return -pi/2;
      }
      else if (x == +inf)
      {
        return value::copysign(T{0}, y);
      }
      else if (x == -inf)
      {
        return value::copysign(pi, y);
      }
    }

    if (x > 0)
    {
      return atan_impl(y / x);
    }
    else if (x < 0)
    {
      if (y > 0) return atan_impl(y / x) + pi;
      else if (y < 0) return atan_impl(y / x) - pi;
      else return value::copysign(pi, y);
    }
    else // if (x == T{0})
    {
      if (y > 0) return pi/2;
      else if (y < 0) return -pi/2;
      else return value::signbit(x) ? value::copysign(pi, y) : value::copysign(T{0}, y);
    }
  }


} // namespace OpenKalman::value::internal


#endif //OPENKALMAN_VALUE_MATH_UTILS_HPP
