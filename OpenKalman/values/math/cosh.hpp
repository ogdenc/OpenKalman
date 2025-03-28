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
 * \brief Definition for \ref value::cosh.
 */

#ifndef OPENKALMAN_VALUE_COSH_HPP
#define OPENKALMAN_VALUE_COSH_HPP

#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/traits/real_type_of_t.hpp"
#include "values/concepts/integral.hpp"
#include "values/classes/operation.hpp"
#include "values/math/real.hpp"
#include "values/math/imag.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "values/math/internal/NaN.hpp"
#include "values/math/internal/infinity.hpp"
#include "values/math/isinf.hpp"
#include "values/math/isnan.hpp"
#include "values/math/exp.hpp"

namespace OpenKalman::value
{
  /**
   * \brief Constexpr alternative to the std::cosh function.
   */
#ifdef __cpp_concepts
  template<value::value Arg>
  constexpr value::value auto cosh(const Arg& arg)
#else
  template<typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
  constexpr auto cosh(const Arg& arg)
#endif
{
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::cosh(a); } };
      return value::operation {Op{}, arg};
    }
    else
    {
      using std::cosh;
      using Return = decltype(cosh(arg));
      struct Op { auto operator()(const Arg& arg) { using std::cosh; return cosh(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return cosh(arg);

      using R = std::conditional_t<value::integral<value::real_type_of_t<Return>>, double, value::real_type_of_t<Return>>;
      if constexpr (value::complex<Return>)
      {
        auto ex = value::exp(value::internal::make_complex_number<R>(arg));
        auto exr = value::real(ex);
        auto exi = value::imag(ex);
        auto half = static_cast<R>(0.5);
        if (exi == 0) return value::internal::make_complex_number<Return>((exr + 1/exr) * half, 0);
        else
        {
          auto denom1 = 1 / (exr*exr + exi*exi);
          return value::internal::make_complex_number<Return>(half * exr * (1 + denom1), half * exi * (1 - denom1));
        }
      }
      else
      {
        if (value::isnan(arg)) return value::internal::NaN<Return>();
        if (value::isinf(arg)) return value::internal::infinity<Return>();
        Return ex = value::exp(arg);
        return (ex + 1/ex) * static_cast<R>(0.5);
      }
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_COSH_HPP
