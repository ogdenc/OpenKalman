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
 * \brief Definition for \ref value::expm1.
 */

#ifndef OPENKALMAN_VALUE_EXPM1_HPP
#define OPENKALMAN_VALUE_EXPM1_HPP

#include <limits>
#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/traits/real_type_of_t.hpp"
#include "values/concepts/integral.hpp"
#include "values/classes/operation.hpp"
#include "real.hpp"
#include "imag.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "internal/NaN.hpp"
#include "internal/infinity.hpp"
#include "isnan.hpp"
#include "internal/periodic_utils.hpp"
#include "internal/exp_utils.hpp"
#include "internal/math_utils.hpp"

namespace OpenKalman::value
{
  /**
   * \brief Constexpr alternative to the std::expm1 function (exponential function minus 1).
   * \details Thus uses a Maclaurin series expansion For floating-point values, or multiplication for integral values.
   * \note This function allows a complex argument, even though the C++ standard does not as of at least C++23.
   */
#ifdef __cpp_concepts
  template<value::value Arg>
  constexpr value::value auto expm1(const Arg& arg)
#else
  template <typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
  constexpr auto expm1(const Arg& arg)
#endif
  {
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::expm1(a); } };
      return value::operation {Op{}, arg};
    }
    else if constexpr (value::complex<Arg>) // Complex expm1 is not defined in the standard.
    {
      using Return = std::decay_t<Arg>;
      using R = std::conditional_t<value::integral<value::real_type_of_t<Return>>, double, value::real_type_of_t<Return>>;
      auto ea = value::expm1(value::real(arg));
      auto b = static_cast<R>(value::imag(arg));
      if (value::isinf(b) or value::isnan(b)) return value::internal::NaN<Return>();
      auto theta{internal::scale_periodic_function(std::move(b))};
      auto sinb = internal::sin_cos_impl<R>(4, theta, theta, theta * theta * theta / -6);
      auto cosbm1 = internal::sin_cos_impl<R>(3, theta, R{0}, static_cast<R>(-0.5) * theta * theta);
      return value::internal::make_complex_number<Return>(ea * (cosbm1 + 1) + cosbm1, ea * sinb + sinb);
    }
    else
    {
      using std::expm1;
      using Return = decltype(expm1(arg));
      struct Op { auto operator()(const Arg& arg) { return expm1(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return expm1(arg);
      else if (value::isnan(arg)) return value::internal::NaN<Return>();
      else if constexpr (value::integral<Arg>)
      {
        return internal::integral_exp<Return>(arg) - Return{1};
      }
      else
      {
        if constexpr (std::numeric_limits<Return>::has_infinity)
        {
          if (arg == std::numeric_limits<Return>::infinity()) return value::internal::infinity<Return>();
          else if (arg == -std::numeric_limits<Return>::infinity()) return Return{-1};
        }

        if (arg >= Return{0} and arg < Return{1})
        {
          if (arg == Return{0}) return arg;
          else return internal::exp_impl<Return>(1, arg, Return{0}, arg);
        }
        else
        {
          int arg_trunc = static_cast<int>(arg) - (arg < Return{0} ? 1 : 0);
          Return arg_frac = static_cast<Return>(arg) - static_cast<Return>(arg_trunc);
          auto et = internal::integral_exp<Return>(arg_trunc) - Return{1};
          auto er = internal::exp_impl<Return>(1, arg_frac, Return{0}, arg_frac);
          return et * er + et + er;
        }
      }
    }
  }

} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_EXPM1_HPP
