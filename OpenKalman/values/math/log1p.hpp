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
 * \brief Definition for \ref values::log1p.
 */

#ifndef OPENKALMAN_VALUE_LOG1P_HPP
#define OPENKALMAN_VALUE_LOG1P_HPP

#include <limits>
#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/traits/real_type_of_t.hpp"
#include "values/classes/operation.hpp"
#include "values/math/real.hpp"
#include "values/math/imag.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "values/math/internal/NaN.hpp"
#include "values/math/internal/infinity.hpp"
#include "values/math/isnan.hpp"
#include "values/math/signbit.hpp"
#include "values/math/copysign.hpp"
#include "internal/math_utils.hpp"
#include "internal/log_utils.hpp"

namespace OpenKalman::values
{
  namespace detail
  {
    // Taylor series for log(1+x)
    template <typename T>
    constexpr T log1p_impl(int n, const T& x, const T& sum, const T& term)
    {
      T next_sum = sum + x * term / n;
      if (sum == next_sum) return sum;
      else return log1p_impl(n + 1, x, next_sum, term * -x);
    }
  }


  /**
   * \brief Constexpr alternative to the std::log1p function, where log1p(x) = log(x+1).
   * \details Unlike the standard function, this one accepts complex arguments.
   */
#ifdef __cpp_concepts
  template<values::value Arg>
  constexpr values::value auto log1p(const Arg& arg)
#else
  template<typename Arg, std::enable_if_t<values::value<Arg>, int> = 0>
  constexpr auto log1p(const Arg& arg)
#endif
  {
    if constexpr (not values::number<Arg>)
    {
      struct Op { constexpr auto operator()(const values::number_type_of_t<Arg>& a) const { return values::log1p(a); } };
      return values::operation {Op{}, arg};
    }
    else if constexpr (values::complex<Arg>)
    {
      using Return = Arg;
      auto re = values::real(values::real(arg));
      auto im = values::real(values::imag(arg));
      using R = decltype(re);
      auto a = static_cast<R>(0.5) * values::log1p(re * re + 2 * re + im * im);
      if constexpr (not std::numeric_limits<values::real_type_of_t<Arg>>::is_iec559) if (values::imag(arg) == 0)
        return values::internal::make_complex_number<Return>(a,
          values::copysign(values::signbit(values::real(arg) + 1) ? numbers::pi_v<R> : 0, values::imag(arg)));
      return values::internal::make_complex_number<Return>(a, internal::atan2_impl(im, re + 1));
    }
    else
    {
      using std::log1p;
      using Return = decltype(log1p(arg));
      struct Op { auto operator()(const Arg& arg) { return log1p(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return log1p(arg);
      if (values::isnan(arg)) return values::internal::NaN<Return>();
      if constexpr (std::numeric_limits<Arg>::has_infinity)
        if (arg == std::numeric_limits<Return>::infinity()) return static_cast<Return>(arg);
      if (arg == 0) return static_cast<Return>(arg);
      if (arg == -1) return -values::internal::infinity<Return>();
      if (arg < -1) return values::internal::NaN<Return>();

      if (static_cast<Return>(-0.125) < arg and arg < static_cast<Return>(0.125))
        return detail::log1p_impl(2, values::real(arg), values::real(arg), -values::real(arg));
      auto [scaled, corr] = arg >= 16 ? internal::log_scaling_gt(values::real(arg) + 1) : internal::log_scaling_lt(values::real(arg) + 1);
      return internal::log_impl(scaled) + corr;
    }
  }


} // namespace OpenKalman::values


#endif //OPENKALMAN_VALUE_LOG1P_HPP
