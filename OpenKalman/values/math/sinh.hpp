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
 * \brief Definition for \ref values::sinh.
 */

#ifndef OPENKALMAN_VALUE_SINH_HPP
#define OPENKALMAN_VALUE_SINH_HPP

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
#include "isinf.hpp"
#include "isnan.hpp"
#include "copysign.hpp"
#include "exp.hpp"
#include "internal/atan_utils.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Constexpr alternative to the std::sinh function.
   */
#ifdef __cpp_concepts
  template<values::value Arg>
  constexpr values::value auto
#else
  template<typename Arg, std::enable_if_t<values::value<Arg>, int> = 0>
  constexpr auto
#endif
  sinh(const Arg& arg)
  {
    if constexpr (not values::number<Arg>)
    {
      struct Op { constexpr auto operator()(const values::number_type_of_t<Arg>& a) const { return values::sinh(a); } };
      return values::operation {Op{}, arg};
    }
    else
    {
      using std::sinh;
      using Return = decltype(sinh(arg));
      struct Op { auto operator()(const Arg& arg) { return sinh(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return sinh(arg);
      else
      {
        using R = std::conditional_t<values::integral<values::real_type_of_t<Return>>, double, values::real_type_of_t<Return>>;
        if constexpr (values::complex<Arg>)
        {
          auto ex = values::exp(values::internal::make_complex_number<R>(arg));
          auto exr = values::real(ex);
          auto exi = values::imag(ex);
          auto half = static_cast<R>(0.5);
          if (exi == 0) return values::internal::make_complex_number<Return>((exr - 1/exr) * half, 0);
          else
          {
            auto denom1 = 1 / (exr*exr + exi*exi);
            return values::internal::make_complex_number<Return>(half * exr * (1 - denom1), half * exi * (1 + denom1));
          }
        }
        else
        {
          if (values::isnan(arg)) return values::internal::NaN<Return>();
          if (arg == 0) return static_cast<Return>(arg);
          if (values::isinf(arg)) return copysign(values::internal::infinity<Return>(), arg);
          Return ex = values::exp(arg);
          return (ex - 1/ex) * static_cast<R>(0.5);
        }
      }
    }
  }


} // namespace OpenKalman::values


#endif //OPENKALMAN_VALUE_SINH_HPP
