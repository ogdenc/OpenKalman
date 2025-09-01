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
 * \brief Definition for \ref values::sin.
 */

#ifndef OPENKALMAN_VALUES_SIN_HPP
#define OPENKALMAN_VALUES_SIN_HPP

#include <limits>
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
#include "values/math/sinh.hpp"
#include "values/math/cosh.hpp"
#include "internal/periodic_utils.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Constexpr alternative to the std::sin function.
   */
#ifdef __cpp_concepts
  template<value Arg>
  constexpr value auto sin(const Arg& arg)
#else
  template<typename Arg, std::enable_if_t<value<Arg>, int> = 0>
  constexpr auto sin(const Arg& arg)
#endif
  {
    if constexpr (fixed<Arg>)
    {
      struct Op { constexpr auto operator()(const value_type_of_t<Arg>& a) const { return values::sin(a); } };
      return values::operation(Op{}, arg);
    }
    else
    {
      using std::sin;
      using Return = decltype(sin(arg));
      struct Op { auto operator()(const Arg& arg) { return sin(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return sin(arg);
      else if constexpr (values::complex<Arg>)
      {
        auto re = values::real(values::real(arg));
        auto im = values::real(values::imag(arg));
        using R = std::decay_t<decltype(re)>;
        if (values::isnan(re) or values::isnan(im)) return values::internal::NaN<Return>();
        auto theta = internal::scale_periodic_function(re);
        auto sinre = internal::sin_cos_impl(4, theta, theta, theta * theta * theta / -6);
        auto cosre = internal::sin_cos_impl(3, theta, R{1}, static_cast<R>(-0.5) * theta * theta);
        return values::internal::make_complex_number<Return>(sinre * values::cosh(im), cosre * values::sinh(im));
      }
      else
      {
        if (values::isinf(arg) or values::isnan(arg)) return values::internal::NaN<Return>();
        auto theta {internal::scale_periodic_function(static_cast<Return>(arg))};
        return internal::sin_cos_impl(4, theta, theta, theta * theta * theta / -6);
      }
    }
  }


}


#endif
