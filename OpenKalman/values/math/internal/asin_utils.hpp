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
 * \internal
 * \brief Utilities for the asin function.
 */

#ifndef OPENKALMAN_VALUES_ASIN_UTILS_HPP
#define OPENKALMAN_VALUES_ASIN_UTILS_HPP

#include "values/math/copysign.hpp"
#include "values/math/sqrt.hpp"

namespace OpenKalman::values::internal
{
 template<typename T>
 constexpr T asin_series(int n, const T& x, const T& sum, const T& term)
 {
  T new_sum {sum + term / static_cast<T>(n)};
  if (sum == new_sum) return sum;
  else return asin_series(n + 2, x, new_sum, term * x * x * static_cast<T>(n)/static_cast<T>(n+1));
 }


 template<typename T>
 constexpr T asin_impl(const T& x)
 {
  T half = static_cast<T>(0.5);
  T pi2 = stdex::numbers::pi_v<T> * half;
  T invsq2 = stdex::numbers::sqrt2_v<T> * half;
  if (-invsq2 <= x and x <= invsq2) return asin_series<T>(3, x, x, half*x*x*x);
  if (invsq2 < x and x < 1) return pi2 - asin_impl(values::sqrt(1 - x*x));
  if (-1 < x and x < -invsq2) return -pi2 + asin_impl(values::sqrt(1 - x*x));
  if (x == 1) return +pi2;
  if (x == -1) return -pi2;
  return values::internal::NaN<T>();
 }
}


#endif
