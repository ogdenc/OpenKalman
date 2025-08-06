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
 * \brief Definition for \ref values::tan.
 */

#ifndef OPENKALMAN_VALUE_TAN_HPP
#define OPENKALMAN_VALUE_TAN_HPP

#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of.hpp"
#include "values/functions/operation.hpp"
#include "values/math/real.hpp"
#include "values/math/imag.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "values/math/internal/NaN.hpp"
#include "values/math/isinf.hpp"
#include "values/math/isnan.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Constexpr alternative to the std::tan function.
   */
#ifdef __cpp_concepts
  template<values::value Arg>
  constexpr values::value auto
#else
  template<typename Arg, std::enable_if_t<values::value<Arg>, int> = 0>
  constexpr auto
#endif
  tan(const Arg& arg)
  {
    if constexpr (fixed<Arg>)
    {
      struct Op { constexpr auto operator()(const number_type_of_t<Arg>& a) const { return values::tan(a); } };
      return values::operation(Op{}, arg);
    }
    else
    {
      using std::tan;
      using Return = decltype(tan(arg));
      struct Op { auto operator()(const Arg& arg) { return tan(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return tan(arg);
      else if constexpr (values::complex<Return>)
      {
        auto sx = values::sin(arg);
        auto cx = values::cos(arg);
        auto sr = values::real(sx);
        auto si = values::imag(sx);
        auto cr = values::real(cx);
        auto ci = values::imag(cx);
        auto denom = cr*cr + ci*ci;
        return values::internal::make_complex_number<Return>((sr*cr + si*ci) / denom, (si*cr - sr*ci) / denom);
      }
      else
      {
        if (values::isnan(arg)) return values::internal::NaN<Return>();
        if (arg == 0) return static_cast<Return>(arg);
        if (values::isinf(arg)) return values::internal::NaN<Return>();
        return static_cast<Return>(values::sin(arg) / values::cos(arg));
      }
    }
  }


} // namespace OpenKalman::values


#endif //OPENKALMAN_VALUE_TAN_HPP
