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
 * \brief Definition for \ref values::exp.
 */

#ifndef OPENKALMAN_VALUE_EXP_HPP
#define OPENKALMAN_VALUE_EXP_HPP

#include <limits>
#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of.hpp"
#include "values/traits/real_type_of.hpp"
#include "values/concepts/integral.hpp"
#include "values/functions/operation.hpp"
#include "real.hpp"
#include "imag.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "internal/NaN.hpp"
#include "internal/infinity.hpp"
#include "isnan.hpp"
#include "internal/periodic_utils.hpp"
#include "internal/exp_utils.hpp"

namespace OpenKalman::values
{
  /**
   * \internal
   * \brief Constexpr alternative to the std::exp function.
   * \details Thus uses a Maclaurin series expansion For floating-point values, or multiplication for integral values.
   */
#ifdef __cpp_concepts
  template<values::value Arg>
  constexpr values::value auto exp(const Arg& arg)
#else
  template<typename Arg, std::enable_if_t<values::value<Arg>, int> = 0>
  constexpr auto exp(const Arg& arg)
#endif
  {
    if constexpr (fixed<Arg>)
    {
      struct Op { constexpr auto operator()(const number_type_of_t<Arg>& a) const { return values::exp(a); } };
      return values::operation(Op{}, arg);
    }
    else
    {
      using std::exp;
      using Return = decltype(exp(arg));
      struct Op { auto operator()(const Arg& arg) { return exp(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return exp(arg);
      else if constexpr (values::integral<Arg>)
      {
        return internal::integral_exp<Return>(arg);
      }
      else if constexpr (values::complex<Arg>)
      {
        using R = real_type_of_t<real_type_of_t<Return>>;
        auto ea = values::exp(values::real(arg));
        auto b = static_cast<R>(values::imag(arg));
        if (values::isinf(b) or values::isnan(b)) return values::internal::NaN<Return>();
        R theta {internal::scale_periodic_function(std::move(b))};
        R sinb = internal::sin_cos_impl<R>(4, theta, theta, theta * theta * theta / -6);
        R cosb = internal::sin_cos_impl<R>(3, theta, R{1}, static_cast<R>(-0.5) * theta * theta);
        return values::internal::make_complex_number<Return>(ea * cosb, ea * sinb);
      }
      else
      {
        if (values::isnan(arg)) return values::internal::NaN<Return>();
        if constexpr (std::numeric_limits<Arg>::has_infinity)
        {
          if (arg == std::numeric_limits<Arg>::infinity()) return values::internal::infinity<Return>();
          else if (arg == -std::numeric_limits<Arg>::infinity()) return Return{0};
        }

        if (arg == Arg{0}) return Return{1};
        else if (arg > Arg{0} and arg < Arg{1})
        {
          return internal::exp_impl(1, static_cast<Return>(arg), Return{1}, static_cast<Return>(arg));
        }
        else
        {
          int arg_trunc = static_cast<int>(arg) - (arg < Arg{0} ? 1 : 0);
          Return arg_frac = static_cast<Return>(arg) - static_cast<Return>(arg_trunc);
          return internal::integral_exp<Return>(arg_trunc) * internal::exp_impl(1, arg_frac, Return{1}, arg_frac);
        }
      }
    }
  }

} // namespace OpenKalman::values


#endif //OPENKALMAN_VALUE_EXP_HPP
