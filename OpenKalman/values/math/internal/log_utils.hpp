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
 * \brief Definition for various utilities for log functions.
 */

#ifndef OPENKALMAN_VALUES_LOG_UTILS_HPP
#define OPENKALMAN_VALUES_LOG_UTILS_HPP

#include "basics/basics.hpp"
#include "values/functions/internal/near.hpp"
#include "values/math/exp.hpp"

namespace OpenKalman::values::internal
{
  // Halley's method
  template <typename T>
  constexpr T log_impl(const T& x, const T& y0 = 0, int cmp = 0)
  {
    auto expy0 = values::exp(y0);
    auto y1 = y0 + T{2} * (x - expy0) / (x + expy0);
    if constexpr (values::complex<T>)
    {
      if (values::internal::near(y1, y0)) return y1;
      else return log_impl(x, y1);
    }
    else
    {
      // Detect when there is a change in direction.
      if (y1 == y0 or (cmp < 0 and y1 > y0) or (cmp > 0 and y1 < y0)) return y1;
      else return log_impl(x, y1, y1 > y0 ? +1 : y1 < y0 ? -1 : 0);
    }
  }


  template <typename T>
  constexpr std::tuple<T, T> log_scaling_gt(const T& x, const T& corr = T{0})
  {
    if (x < T{0x1p+4}) return {x, corr};
    else if (x < T{0x1p+16}) return log_scaling_gt<T>(x * T{0x1p-4}, corr + T{4} * stdcompat::numbers::ln2_v<T>);
    else if (x < T{0x1p+64}) return log_scaling_gt<T>(x * T{0x1p-16}, corr + T{16} * stdcompat::numbers::ln2_v<T>);
    else return log_scaling_gt<T>(x * T{0x1p-64}, corr + T{64} * stdcompat::numbers::ln2_v<T>);
  }


  template <typename T>
  constexpr std::tuple<T, T> log_scaling_lt(const T& x, const T& corr = T{0})
  {
    if (x > T{0x1p-4}) return {x, corr};
    else if (x > T{0x1p-16}) return log_scaling_lt<T>(x * T{0x1p+4}, corr - T{4} * stdcompat::numbers::ln2_v<T>);
    else if (x > T{0x1p-64}) return log_scaling_lt<T>(x * T{0x1p+16}, corr - T{16} * stdcompat::numbers::ln2_v<T>);
    else return log_scaling_lt<T>(x * T{0x1p+64}, corr - T{64} * stdcompat::numbers::ln2_v<T>);
  }


}


#endif
