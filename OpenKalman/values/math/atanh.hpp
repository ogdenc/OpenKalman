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
 * \brief Definition for \ref values::atanh.
 */

#ifndef OPENKALMAN_VALUE_ATANH_HPP
#define OPENKALMAN_VALUE_ATANH_HPP

#include <limits>
#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/classes/operation.hpp"
#include "values/math/real.hpp"
#include "values/math/imag.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "values/math/internal/NaN.hpp"
#include "values/math/internal/infinity.hpp"
#include "values/math/isnan.hpp"
#include "values/math/log.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Constexpr alternative to the std::atanh function.
   */
#ifdef __cpp_concepts
  template<values::value Arg>
  constexpr values::value auto atanh(const Arg& arg)
#else
  template<typename Arg, std::enable_if_t<values::value<Arg>, int> = 0>
  constexpr auto atanh(const Arg& arg)
#endif
  {
    if constexpr (not values::number<Arg>)
    {
      struct Op { constexpr auto operator()(const values::number_type_of_t<Arg>& a) const { return values::atanh(a); } };
      return values::operation {Op{}, arg};
    }
    else
    {
      using std::atanh;
      using Return = std::decay_t<decltype(atanh(arg))>;
      struct Op { auto operator()(const Arg& arg) { return atanh(arg); } };
      if (values::internal::constexpr_callable<Op>(arg)) return atanh(arg);
      else if constexpr (values::complex<Return>)
      {
        auto xr = values::real(values::real(arg));
        auto xi = values::real(values::imag(arg));
        using R = std::decay_t<decltype(xr)>;
        auto denom = 1 - 2*xr + xr*xr + xi*xi;
        auto lg = values::log(values::internal::make_complex_number((1 - xr*xr - xi*xi) / denom, 2 * xi / denom));
        auto half = static_cast<R>(0.5);
        return values::internal::make_complex_number<Return>(half * values::real(lg), half * values::imag(lg));
      }
      else
      {
        if (values::isnan(arg)) return values::internal::NaN<Return>();
        if constexpr (std::numeric_limits<Return>::has_infinity)
        {
          if (arg < -1 or arg > 1) return values::internal::NaN<Return>();
          else if (arg == 1) return values::internal::infinity<Return>();
          else if (arg == -1) return -values::internal::infinity<Return>();
        }
        if (arg == 0) return static_cast<Return>(arg);
        auto x = values::real(arg);
        return static_cast<Return>(0.5) * values::log((1 + x) / (1 - x));
      }
    }
  }


} // namespace OpenKalman::values


#endif //OPENKALMAN_VALUE_ATANH_HPP
