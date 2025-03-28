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
 * \brief Definition of utilities for atan functions.
 */

#ifndef OPENKALMAN_VALUE_ATAN_UTILS_HPP
#define OPENKALMAN_VALUE_ATAN_UTILS_HPP

#include <type_traits>
#include <limits>
#include "values/math/real.hpp"
#include "values/math/imag.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/math/signbit.hpp"
#include "values/math/copysign.hpp"
#include "values/math/log1p.hpp"

namespace OpenKalman::value::internal
{
  template <typename T>
  constexpr T atan_impl_general(const T& x)
  {
    if constexpr (value::complex<T>)
    {
      using R = std::decay_t<decltype(value::real(x))>;

      auto xr = value::real(x);
      auto xi = value::imag(x);
      auto ar = -xi;
      auto ai = xr;
      auto half = static_cast<R>(0.5);
      auto lar = half * value::log1p(ar * ar + 2 * ar + ai * ai);
      auto lai = not std::numeric_limits<R>::is_iec559 and ai == 0 ?
        value::copysign(value::signbit(ar + 1) ? numbers::pi_v<R> : 0, ai) : atan2_impl(ai, ar + 1);
      auto br = xi;
      auto bi = -xr;
      auto lbr = half * value::log1p(br * br + 2 * br + bi * bi);
      auto lbi = not std::numeric_limits<R>::is_iec559 and bi == 0 ?
        value::copysign(value::signbit(br + 1) ? numbers::pi_v<R> : 0, bi) : atan2_impl(bi, br + 1);
      return value::internal::make_complex_number<T>(half * (lai - lbi), half * (lbr - lar));
    }
    else return atan_impl(x);
  }


} // namespace OpenKalman::value::internal


#endif //OPENKALMAN_VALUE_ATAN_UTILS_HPP
