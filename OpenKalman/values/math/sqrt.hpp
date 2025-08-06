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
 * \brief Definition for \ref values::sqrt.
 */

#ifndef OPENKALMAN_VALUE_SQRT_HPP
#define OPENKALMAN_VALUE_SQRT_HPP

#include <limits>
#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of.hpp"
#include "values/traits/real_type_of.hpp"
#include "values/functions/operation.hpp"
#include "values/math/real.hpp"
#include "values/math/imag.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "values/math/internal/NaN.hpp"
#include "values/math/internal/infinity.hpp"
#include "values/math/isinf.hpp"
#include "values/math/isnan.hpp"

#include "values/math/copysign.hpp"

namespace OpenKalman::values
{
  /**
   * \brief A constexpr alternative to std::sqrt.
   * \details Uses the Newton-Raphson method
   */
#ifdef __cpp_concepts
  template<value Arg>
  constexpr value auto sqrt(const Arg& arg)
#else
  template <typename Arg, std::enable_if_t<value<Arg>, int> = 0>
  constexpr auto sqrt(const Arg& arg)
#endif
  {
    if constexpr (fixed<Arg>)
    {
      struct Op { constexpr auto operator()(const number_type_of_t<Arg>& a) const { return values::sqrt(a); } };
      return values::operation(Op{}, arg);
    }
    else
    {
      using std::sqrt;
      using Return = std::decay_t<decltype(sqrt(arg))>;
      struct Op { auto operator()(const Arg& arg) { return sqrt(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return sqrt(arg);
      else if constexpr (values::complex<Arg>)
      {
        // Find the principal square root
        auto arg_re = values::real(arg);
        auto arg_im = values::imag(arg);
        using R = real_type_of_t<real_type_of_t<Return>>;
        if constexpr (std::numeric_limits<real_type_of_t<Arg>>::is_iec559)
        {
          if (values::isinf(arg_im)) return values::internal::make_complex_number<Return>(
            values::internal::infinity<R>(),
            values::copysign(values::internal::infinity<R>(), arg_im));
          if (arg_re == values::internal::infinity<real_type_of_t<Arg>>())
            return values::internal::make_complex_number<Return>(
              values::internal::infinity<R>(),
              values::copysign(values::isnan(arg_im) ? values::internal::NaN<R>() : 0, arg_im));
          if (arg_re == -values::internal::infinity<real_type_of_t<Arg>>())
            return values::internal::make_complex_number<Return>(
              values::isnan(arg_im) ? values::internal::NaN<R>() : 0,
              values::copysign(values::internal::infinity<R>(), arg_im));
        }
        auto re = values::real(arg_re);
        auto im = values::real(arg_im);
        auto nx = values::sqrt(re * re + im * im);
        auto half = static_cast<std::decay_t<decltype(re)>>(0.5);
        auto sqp = values::sqrt(half * (nx + re));
        auto sqm = values::sqrt(half * (nx - re));
        return values::internal::make_complex_number<Return>(sqp, values::copysign(sqm, im));
      }
      else
      {
        if (values::isnan(arg) or arg == 0) return static_cast<Return>(arg); //< This will potentially preserve the sign.
        if (arg < 0) return values::internal::NaN<Return>();
        if constexpr (std::numeric_limits<Arg>::has_infinity)
          if (arg == values::internal::infinity<Arg>()) return values::internal::infinity<Return>();
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

} // namespace OpenKalman::values


#endif //OPENKALMAN_VALUE_SQRT_HPP
