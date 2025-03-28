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
 * \brief Definition for \ref value::atanh.
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

namespace OpenKalman::value
{
  /**
   * \brief Constexpr alternative to the std::atanh function.
   */
#ifdef __cpp_concepts
  template<value::value Arg>
  constexpr value::value auto atanh(const Arg& arg)
#else
  template<typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
  constexpr auto atanh(const Arg& arg)
#endif
  {
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::atanh(a); } };
      return value::operation {Op{}, arg};
    }
    else
    {
      using std::atanh;
      using Return = std::decay_t<decltype(atanh(arg))>;
      struct Op { auto operator()(const Arg& arg) { return atanh(arg); } };
      if (value::internal::constexpr_callable<Op>(arg)) return atanh(arg);
      else if constexpr (value::complex<Return>)
      {
        auto xr = value::real(value::real(arg));
        auto xi = value::real(value::imag(arg));
        using R = std::decay_t<decltype(xr)>;
        auto denom = 1 - 2*xr + xr*xr + xi*xi;
        auto lg = value::log(value::internal::make_complex_number((1 - xr*xr - xi*xi) / denom, 2 * xi / denom));
        auto half = static_cast<R>(0.5);
        return value::internal::make_complex_number<Return>(half * value::real(lg), half * value::imag(lg));
      }
      else
      {
        if (value::isnan(arg)) return value::internal::NaN<Return>();
        if constexpr (std::numeric_limits<Return>::has_infinity)
        {
          if (arg < -1 or arg > 1) return value::internal::NaN<Return>();
          else if (arg == 1) return value::internal::infinity<Return>();
          else if (arg == -1) return -value::internal::infinity<Return>();
        }
        if (arg == 0) return static_cast<Return>(arg);
        auto x = value::real(arg);
        return static_cast<Return>(0.5) * value::log((1 + x) / (1 - x));
      }
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_ATANH_HPP
