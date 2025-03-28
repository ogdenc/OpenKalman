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
 * \brief Definition for \ref value::tan.
 */

#ifndef OPENKALMAN_VALUE_TAN_HPP
#define OPENKALMAN_VALUE_TAN_HPP

#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/classes/operation.hpp"
#include "values/math/real.hpp"
#include "values/math/imag.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "values/math/internal/NaN.hpp"
#include "values/math/isinf.hpp"
#include "values/math/isnan.hpp"

namespace OpenKalman::value
{
  /**
   * \brief Constexpr alternative to the std::tan function.
   */
#ifdef __cpp_concepts
  template<value::value Arg>
  constexpr value::value auto
#else
  template<typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
  constexpr auto
#endif
  tan(const Arg& arg)
  {
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::tan(a); } };
      return value::operation {Op{}, arg};
    }
    else
    {
      using std::tan;
      using Return = decltype(tan(arg));
      struct Op { auto operator()(const Arg& arg) { return tan(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return tan(arg);
      else if constexpr (value::complex<Return>)
      {
        auto sx = value::sin(arg);
        auto cx = value::cos(arg);
        auto sr = value::real(sx);
        auto si = value::imag(sx);
        auto cr = value::real(cx);
        auto ci = value::imag(cx);
        auto denom = cr*cr + ci*ci;
        return value::internal::make_complex_number<Return>((sr*cr + si*ci) / denom, (si*cr - sr*ci) / denom);
      }
      else
      {
        if (value::isnan(arg)) return value::internal::NaN<Return>();
        if (arg == 0) return static_cast<Return>(arg);
        if (value::isinf(arg)) return value::internal::NaN<Return>();
        return static_cast<Return>(value::sin(arg) / value::cos(arg));
      }
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_TAN_HPP
