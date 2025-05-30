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
 * \brief Definition for \ref values::cos.
 */

#ifndef OPENKALMAN_VALUE_COS_HPP
#define OPENKALMAN_VALUE_COS_HPP

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

namespace OpenKalman::values
{
  /**
   * \brief Constexpr alternative to the std::cos function.
   */
#ifdef __cpp_concepts
template<values::value Arg>
  constexpr values::value auto cos(const Arg& arg)
#else
  template<typename Arg>
  constexpr auto cos(const Arg& arg, std::enable_if_t<values::value<Arg>, int> = 0)
#endif
  {
    if constexpr (not values::number<Arg>)
    {
      struct Op { constexpr auto operator()(const values::number_type_of_t<Arg>& a) const { return values::cos(a); } };
      return values::operation {Op{}, arg};
    }
    else
    {
      using std::cos;
      using Return = decltype(cos(arg));
      struct Op { auto operator()(const Arg& arg) { return cos(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return cos(arg);
      else if constexpr (values::complex<Arg>)
      {
        auto re = values::real(values::real(arg));
        auto im = values::real(values::imag(arg));
        if (values::isnan(re) or values::isnan(im)) return values::internal::NaN<Return>();
        using R = std::decay_t<decltype(re)>;
        if (values::isinf(re) or values::isnan(re)) return values::internal::NaN<Return>();
        auto theta = internal::scale_periodic_function(re);
        auto sinre = internal::sin_cos_impl(4, theta, theta, theta * theta * theta / -6);
        auto cosre = internal::sin_cos_impl(3, theta, R{1}, static_cast<R>(-0.5) * theta * theta);
        return values::internal::make_complex_number<Return>(cosre * values::cosh(im), -sinre * values::sinh(im));
      }
      else
      {
        if (values::isinf(arg) or values::isnan(arg)) return values::internal::NaN<Return>();
        auto theta{internal::scale_periodic_function(static_cast<Return>(arg))};
        return internal::sin_cos_impl(3, theta, Return{1}, static_cast<Return>(-0.5) * theta * theta);
      }
    }
  }


} // namespace OpenKalman::values


#endif //OPENKALMAN_VALUE_COS_HPP
