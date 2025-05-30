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
 * \brief Definition for \ref values::acosh.
 */

#ifndef OPENKALMAN_VALUE_ACOSH_HPP
#define OPENKALMAN_VALUE_ACOSH_HPP

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
#include "values/math/sqrt.hpp"
#include "values/math/log.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Constexpr alternative to the std::acosh function.
   */
#ifdef __cpp_concepts
  template<values::value Arg>
  constexpr values::value auto acosh(const Arg& arg)
#else
  template<typename Arg, std::enable_if_t<values::value<Arg>, int> = 0>
  constexpr auto acosh(const Arg& arg)
#endif
  {
    if constexpr (not values::number<Arg>)
    {
      struct Op { constexpr auto operator()(const values::number_type_of_t<Arg>& a) const { return values::acosh(a); } };
      return values::operation {Op{}, arg};
    }
    else
    {
      using std::acosh;
      using Return = std::decay_t<decltype(acosh(arg))>;
      struct Op { auto operator()(const Arg& arg) { return acosh(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return acosh(arg);
      else if constexpr (values::complex<Return>)
      {
        auto xr = values::real(values::real(arg));
        auto xi = values::real(values::imag(arg));
        using R = std::decay_t<decltype(xr)>;
        auto sqtp = values::sqrt(values::internal::make_complex_number<R>(xr + 1, xi));
        auto a = values::real(sqtp);
        auto b = values::imag(sqtp);
        auto sqtm = values::sqrt(values::internal::make_complex_number<R>(xr - 1, xi));
        auto c = values::real(sqtm);
        auto d = values::imag(sqtm);
        return values::internal::make_complex_number<Return>(values::log(values::internal::make_complex_number(xr + a*c - b*d, xi + a*d + b*c)));
      }
      else
      {
        if (values::isnan(arg)) return values::internal::NaN<Return>();
        if (arg == 1) return static_cast<Return>(+0.);
        if constexpr (std::numeric_limits<Return>::has_infinity) if (arg == values::internal::infinity<Return>())
          return static_cast<Return>(arg);
        if (arg < 1) return values::internal::NaN<Return>();
        auto x = values::real(arg);
        return values::log(x + values::sqrt(x * x - 1));
      }
    }
  }


} // namespace OpenKalman::values


#endif //OPENKALMAN_VALUE_ACOSH_HPP
