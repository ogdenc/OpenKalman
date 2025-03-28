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
 * \brief Definition for \ref value::log1p.
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

namespace OpenKalman::value
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
  template<value::value Arg>
  constexpr value::value auto log1p(const Arg& arg)
#else
  template<typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
  constexpr auto log1p(const Arg& arg)
#endif
  {
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::log1p(a); } };
      return value::operation {Op{}, arg};
    }
    else if constexpr (value::complex<Arg>)
    {
      using Return = Arg;
      auto re = value::real(value::real(arg));
      auto im = value::real(value::imag(arg));
      using R = decltype(re);
      auto a = static_cast<R>(0.5) * value::log1p(re * re + 2 * re + im * im);
      if constexpr (not std::numeric_limits<value::real_type_of_t<Arg>>::is_iec559) if (value::imag(arg) == 0)
        return value::internal::make_complex_number<Return>(a,
          value::copysign(value::signbit(value::real(arg) + 1) ? numbers::pi_v<R> : 0, value::imag(arg)));
      return value::internal::make_complex_number<Return>(a, internal::atan2_impl(im, re + 1));
    }
    else
    {
      using std::log1p;
      using Return = decltype(log1p(arg));
      struct Op { auto operator()(const Arg& arg) { return log1p(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return log1p(arg);
      if (value::isnan(arg)) return value::internal::NaN<Return>();
      if constexpr (std::numeric_limits<Arg>::has_infinity)
        if (arg == std::numeric_limits<Return>::infinity()) return static_cast<Return>(arg);
      if (arg == 0) return static_cast<Return>(arg);
      if (arg == -1) return -value::internal::infinity<Return>();
      if (arg < -1) return value::internal::NaN<Return>();

      if (static_cast<Return>(-0.125) < arg and arg < static_cast<Return>(0.125))
        return detail::log1p_impl(2, value::real(arg), value::real(arg), -value::real(arg));
      auto [scaled, corr] = arg >= 16 ? internal::log_scaling_gt(value::real(arg) + 1) : internal::log_scaling_lt(value::real(arg) + 1);
      return internal::log_impl(scaled) + corr;
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_LOG1P_HPP
