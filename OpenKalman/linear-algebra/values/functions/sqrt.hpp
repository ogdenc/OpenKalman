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
 * \brief Definition for \ref value::sqrt.
 */

#ifndef OPENKALMAN_VALUE_SQRT_HPP
#define OPENKALMAN_VALUE_SQRT_HPP

#include <limits>
#include "linear-algebra/values/interface/number_traits.hpp"
#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/values/concepts/value.hpp"
#include "linear-algebra/values/traits/number_type_of_t.hpp"
#include "linear-algebra/values/traits/real_type_of_t.hpp"
#include "linear-algebra/values/classes/operation.hpp"
#include "linear-algebra/values/functions/real.hpp"
#include "linear-algebra/values/functions/imag.hpp"
#include "linear-algebra/values/functions/internal/make_complex_number.hpp"
#include "linear-algebra/values/functions/internal/constexpr_callable.hpp"
#include "linear-algebra/values/functions/internal/NaN.hpp"
#include "linear-algebra/values/functions/internal/infinity.hpp"
#include "linear-algebra/values/functions/isinf.hpp"
#include "linear-algebra/values/functions/isnan.hpp"

#include "linear-algebra/values/functions/copysign.hpp"

namespace OpenKalman::value
{
  /**
   * \brief A constexpr alternative to std::sqrt.
   * \details Uses the Newton-Raphson method
   */
#ifdef __cpp_concepts
  template<value::value Arg>
  constexpr value::value auto sqrt(const Arg& arg)
#else
  template <typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
  constexpr auto sqrt(const Arg& arg)
#endif
  {
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::sqrt(a); } };
      return value::operation {Op{}, arg};
    }
    else
    {
      using std::sqrt;
      using Return = std::decay_t<decltype(sqrt(arg))>;
      struct Op { constexpr auto operator()(const Arg& arg) { return sqrt(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return sqrt(arg);
      else if constexpr (value::complex<Arg>)
      {
        // Find the principal square root
        auto arg_re = value::real(arg);
        auto arg_im = value::imag(arg);
        using R = real_type_of_t<Arg>;
        if constexpr (std::numeric_limits<R>::is_iec559)
        {
          if (value::isinf(arg_im)) return value::internal::make_complex_number<Return>(
            value::internal::infinity<R>(),
            value::copysign(value::internal::infinity<R>(), arg_im));
          if (arg_re == value::internal::infinity<R>()) return value::internal::make_complex_number<Return>(
            value::internal::infinity<R>(),
            value::copysign(value::isnan(arg_im) ? value::internal::NaN<R>() : 0, arg_im));
          if (arg_re == -value::internal::infinity<R>()) return value::internal::make_complex_number<Return>(
            value::isnan(arg_im) ? value::internal::NaN<R>() : 0,
            value::copysign(value::internal::infinity<R>(), arg_im));
        }
        auto re = value::real(arg_re);
        auto im = value::real(arg_im);
        auto nx = value::sqrt(re * re + im * im);
        auto half = static_cast<std::decay_t<decltype(re)>>(0.5);
        auto sqp = value::sqrt(half * (nx + re));
        auto sqm = value::sqrt(half * (nx - re));
        return value::internal::make_complex_number<Return>(sqp, value::copysign(sqm, im));
      }
      else
      {
        if (value::isnan(arg) or arg == 0) return static_cast<Return>(arg); //< This will potentially preserve the sign.
        if (arg < 0) return value::internal::NaN<Return>();
        if constexpr (std::numeric_limits<Arg>::has_infinity)
          if (arg == value::internal::infinity<Arg>()) return value::internal::infinity<Return>();
        auto half = static_cast<Return>(0.5);
        Return next = half * static_cast<Return>(arg);
        Return previous = 0;
        while (next != previous)
        {
          previous = next;
          next = half * (previous + static_cast<Return>(arg) / previous);
        }
        return next;
      }

      /** // Code for a purely integral version:
      T lo = 0 , hi = x / 2 + 1;
      while (lo != hi) { const T mid = (lo + hi + 1) / 2; if (x / mid < mid) hi = mid - 1; else lo = mid; }
      return lo;*/
    }
  }

} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_SQRT_HPP
