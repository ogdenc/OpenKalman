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
#include "../../../basics/compatibility/language-features.hpp"
#include "values/math/signbit.hpp"
#include "values/math/copysign.hpp"
#include "values/math/sqrt.hpp"
#include "asin_utils.hpp"

namespace OpenKalman::values::internal
{
  template<typename T>
  constexpr T atan_impl(const T& x)
  {
    return asin_impl(x / values::sqrt(1 + x * x));
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
        return values::copysign(T{0}, y);
      }
      else if (x == -inf)
      {
        return values::copysign(pi, y);
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
      else return values::copysign(pi, y);
    }
    else // if (x == T{0})
    {
      if (y > 0) return pi/2;
      else if (y < 0) return -pi/2;
      else return values::signbit(x) ? values::copysign(pi, y) : values::copysign(T{0}, y);
    }
  }


} // namespace OpenKalman::values::internal


#endif //OPENKALMAN_VALUE_MATH_UTILS_HPP
