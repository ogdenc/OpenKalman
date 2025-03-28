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
 * \brief Definition for \ref value::sinh.
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

namespace OpenKalman::value
{
  /**
   * \brief Constexpr alternative to the std::sinh function.
   */
#ifdef __cpp_concepts
  template<value::value Arg>
  constexpr value::value auto
#else
  template<typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
  constexpr auto
#endif
  sinh(const Arg& arg)
  {
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::sinh(a); } };
      return value::operation {Op{}, arg};
    }
    else
    {
      using std::sinh;
      using Return = decltype(sinh(arg));
      struct Op { auto operator()(const Arg& arg) { return sinh(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return sinh(arg);
      else
      {
        using R = std::conditional_t<value::integral<value::real_type_of_t<Return>>, double, value::real_type_of_t<Return>>;
        if constexpr (value::complex<Arg>)
        {
          auto ex = value::exp(value::internal::make_complex_number<R>(arg));
          auto exr = value::real(ex);
          auto exi = value::imag(ex);
          auto half = static_cast<R>(0.5);
          if (exi == 0) return value::internal::make_complex_number<Return>((exr - 1/exr) * half, 0);
          else
          {
            auto denom1 = 1 / (exr*exr + exi*exi);
            return value::internal::make_complex_number<Return>(half * exr * (1 - denom1), half * exi * (1 + denom1));
          }
        }
        else
        {
          if (value::isnan(arg)) return value::internal::NaN<Return>();
          if (arg == 0) return static_cast<Return>(arg);
          if (value::isinf(arg)) return copysign(value::internal::infinity<Return>(), arg);
          Return ex = value::exp(arg);
          return (ex - 1/ex) * static_cast<R>(0.5);
        }
      }
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_SINH_HPP
