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
 * \brief Definition for \ref value::tanh.
 */

#ifndef OPENKALMAN_VALUE_TANH_HPP
#define OPENKALMAN_VALUE_TANH_HPP

#include <limits>
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
#include "values/math/isnan.hpp"
#include "values/math/exp.hpp"

namespace OpenKalman::value
{
  /**
   * \internal
   * \brief Constexpr alternative to the std::tanh function.
   */
#ifdef __cpp_concepts
  template<value::value Arg>
  constexpr value::value auto tanh(const Arg& arg)
#else
  template<typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
  constexpr auto tanh(const Arg& arg)
#endif
  {
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::tanh(a); } };
      return value::operation {Op{}, arg};
    }
    else
    {
      using std::tanh;
      using Return = decltype(tanh(arg));
      struct Op { auto operator()(const Arg& arg) { return tanh(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return tanh(arg);
      else if constexpr (value::complex<Return>)
      {
        if (arg == Arg{0}) return static_cast<Return>(arg);
        if (value::real(arg) == 0 and value::imag(arg) == 0) return value::internal::make_complex_number<Return>(0, 0);
        using R = std::conditional_t<value::integral<value::real_type_of_t<Return>>, double, value::real_type_of_t<Return>>;
        auto ex = value::exp(value::internal::make_complex_number<R>(arg));
        auto er = value::real(ex);
        auto ei = value::imag(ex);
        auto d = er*er - ei*ei;
        auto b = 2*er*ei;
        auto d2b2 = d*d + b*b;
        auto denom = R{d2b2 + 2*d + 1};
        return value::internal::make_complex_number<Return>(R{d2b2 - 1} / denom, R{2 * b} / denom);
      }
      else
      {
        if (value::isnan(arg)) return value::internal::NaN<Return>();
        if constexpr (std::numeric_limits<Arg>::has_infinity)
        {
          if (arg == std::numeric_limits<Arg>::infinity()) return Return{1};
          else if (arg == -std::numeric_limits<Arg>::infinity()) return Return{-1};
        }
        if (arg == Arg{0}) return static_cast<Return>(arg);

        Return ex = value::exp(arg);
        Return ex2 = ex * ex;
        return (ex2 - Return{1}) / (ex2 + Return{1});
      }
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_TANH_HPP
