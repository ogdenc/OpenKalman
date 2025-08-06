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
 * \brief Definition for periodic-function utilities for math functions.
 */

#ifndef OPENKALMAN_VALUE_PERIODIC_UTILS_HPP
#define OPENKALMAN_VALUE_PERIODIC_UTILS_HPP

#include <cstdint>
#include <limits>
#include "basics/basics.hpp"

namespace OpenKalman::values::internal
{
  // Taylor series expansion
  template <typename T>
  constexpr T sin_cos_impl(int i, const T& x, const T& sum, const T& term)
  {
    auto new_sum = sum + term;
    //if (values::internal::near(sum, new_sum)) return new_sum;
    if (sum == new_sum) return sum;
    else return sin_cos_impl(i + 2, x, new_sum, term * x * x / static_cast<T>(-i * (i + 1)));
  }


  // Scale a periodic function (e.g., sin or cos) to within ±pi
  template <typename T>
  constexpr T scale_periodic_function(const T& theta)
  {
    T pi2 {stdcompat::numbers::pi_v<T> * 2};
    T max {static_cast<T>(std::numeric_limits<std::intmax_t>::max())};
    T lowest {static_cast<T>(std::numeric_limits<std::intmax_t>::lowest())};
    if (theta > -pi2 and theta < pi2)
    {
      return theta;
    }
    else if (theta / pi2 >= lowest and theta / pi2 <= max)
    {
      return theta - static_cast<std::intmax_t>(theta / pi2) * pi2;
    }
    else if (theta > 0)
    {
      T corr {pi2};
      while ((theta - corr) / pi2 > max) corr *= 2;
      return scale_periodic_function(theta - corr);
    }
    else
    {
      T corr {pi2};
      while ((theta + corr) / pi2 < lowest) corr *= 2;
      return scale_periodic_function(theta + corr);
    }
  }


} // namespace OpenKalman::values::internal


#endif //OPENKALMAN_VALUE_PERIODIC_UTILS_HPP
