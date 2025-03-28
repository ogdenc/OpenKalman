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
 * \brief Definition for \ref value::sin.
 */

#ifndef OPENKALMAN_VALUE_SIN_HPP
#define OPENKALMAN_VALUE_SIN_HPP

#include <limits>
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
#include "values/math/sinh.hpp"
#include "values/math/cosh.hpp"
#include "internal/periodic_utils.hpp"

namespace OpenKalman::value
{
  /**
   * \brief Constexpr alternative to the std::sin function.
   */
#ifdef __cpp_concepts
  template<value::value Arg>
  constexpr value::value auto sin(const Arg& arg)
#else
  template<typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
  constexpr auto sin(const Arg& arg)
#endif
  {
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::sin(a); } };
      return value::operation {Op{}, arg};
    }
    else
    {
      using std::sin;
      using Return = decltype(sin(arg));
      struct Op { auto operator()(const Arg& arg) { return sin(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return sin(arg);
      else if constexpr (value::complex<Arg>)
      {
        auto re = value::real(value::real(arg));
        auto im = value::real(value::imag(arg));
        using R = std::decay_t<decltype(re)>;
        if (value::isnan(re) or value::isnan(im)) return value::internal::NaN<Return>();
        auto theta = internal::scale_periodic_function(re);
        auto sinre = internal::sin_cos_impl(4, theta, theta, theta * theta * theta / -6);
        auto cosre = internal::sin_cos_impl(3, theta, R{1}, static_cast<R>(-0.5) * theta * theta);
        return value::internal::make_complex_number<Return>(sinre * value::cosh(im), cosre * value::sinh(im));
      }
      else
      {
        if (value::isinf(arg) or value::isnan(arg)) return value::internal::NaN<Return>();
        auto theta {internal::scale_periodic_function(static_cast<Return>(arg))};
        return internal::sin_cos_impl(4, theta, theta, theta * theta * theta / -6);
      }
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_SIN_HPP
