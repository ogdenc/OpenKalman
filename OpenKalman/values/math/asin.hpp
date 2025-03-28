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
 * \brief Definition for \ref value::asin.
 */

#ifndef OPENKALMAN_VALUE_ASIN_HPP
#define OPENKALMAN_VALUE_ASIN_HPP

#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/classes/operation.hpp"
#include "values/math/real.hpp"
#include "values/math/imag.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "values/math/internal/NaN.hpp"
#include "values/math/isnan.hpp"
#include "values/math/sqrt.hpp"
#include "values/math/internal/asin_utils.hpp"
#include "values/math/log.hpp"

namespace OpenKalman::value
{
  /**
   * \brief Constexpr alternative to the std::asin function.
   */
#ifdef __cpp_concepts
  template<value::value Arg>
  constexpr value::value auto
#else
  template<typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
  constexpr auto
#endif
  asin(const Arg& arg)
  {
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::asin(a); } };
      return value::operation {Op{}, arg};
    }
    else
    {
      using std::asin;
      using Return = std::decay_t<decltype(asin(arg))>;
      struct Op { auto operator()(const Arg& arg) { return asin(arg); } };
      if (value::internal::constexpr_callable<Op>(arg)) return asin(arg);
      else if constexpr (value::complex<Return>)
      {
        auto xr = value::real(value::real(arg));
        auto xi = value::real(value::imag(arg));
        auto sqt = value::sqrt(value::internal::make_complex_number(1 - xr*xr + xi*xi, - 2*xr*xi));
        auto lg = value::log(value::internal::make_complex_number(value::real(sqt) + xi, value::imag(sqt) - xr));
        return value::internal::make_complex_number<Return>(-value::imag(lg), value::real(lg));
      }
      else
      {
        if (value::isnan(arg)) return value::internal::NaN<Return>();
        return internal::asin_impl(static_cast<Return>(arg));
      }
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_ASIN_HPP
