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
 * \brief Definition for \ref values::asin.
 */

#ifndef OPENKALMAN_VALUES_ASIN_HPP
#define OPENKALMAN_VALUES_ASIN_HPP

#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/value_type_of.hpp"
#include "values/functions/operation.hpp"
#include "values/math/real.hpp"
#include "values/math/imag.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "values/math/internal/NaN.hpp"
#include "values/math/isnan.hpp"
#include "values/math/sqrt.hpp"
#include "values/math/internal/asin_utils.hpp"
#include "values/math/log.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Constexpr alternative to the std::asin function.
   */
#ifdef __cpp_concepts
  template<value Arg>
  constexpr value auto
#else
  template<typename Arg, std::enable_if_t<value<Arg>, int> = 0>
  constexpr auto
#endif
  asin(const Arg& arg)
  {
    if constexpr (fixed<Arg>)
    {
      struct Op { constexpr auto operator()(const value_type_of_t<Arg>& a) const { return values::asin(a); } };
      return values::operation(Op{}, arg);
    }
    else
    {
      using std::asin;
      using Return = std::decay_t<decltype(asin(arg))>;
      struct Op { auto operator()(const Arg& arg) { return asin(arg); } };
      if (values::internal::constexpr_callable<Op>(arg)) return asin(arg);
      else if constexpr (values::complex<Return>)
      {
        auto xr = values::real(values::real(arg));
        auto xi = values::real(values::imag(arg));
        auto sqt = values::sqrt(values::internal::make_complex_number<>(1 - xr*xr + xi*xi, - 2*xr*xi));
        auto lg = values::log(values::internal::make_complex_number<>(values::real(sqt) + xi, values::imag(sqt) - xr));
        return values::internal::make_complex_number<Return>(-values::imag(lg), values::real(lg));
      }
      else
      {
        if (values::isnan(arg)) return values::internal::NaN<Return>();
        return internal::asin_impl(static_cast<Return>(arg));
      }
    }
  }


}


#endif
