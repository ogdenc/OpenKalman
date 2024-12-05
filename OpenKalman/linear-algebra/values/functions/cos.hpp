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
 * \brief Definition for \ref value::cos.
 */

#ifndef OPENKALMAN_VALUE_COS_HPP
#define OPENKALMAN_VALUE_COS_HPP

#include <limits>
#include "linear-algebra/values/interface/number_traits.hpp"
#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/values/concepts/value.hpp"
#include "linear-algebra/values/traits/number_type_of_t.hpp"
#include "linear-algebra/values/classes/operation.hpp"
#include "linear-algebra/values/functions/real.hpp"
#include "linear-algebra/values/functions/imag.hpp"
#include "linear-algebra/values/functions/internal/make_complex_number.hpp"
#include "linear-algebra/values/functions/internal/constexpr_callable.hpp"
#include "linear-algebra/values/functions/internal/NaN.hpp"
#include "linear-algebra/values/functions/isnan.hpp"
#include "linear-algebra/values/functions/sinh.hpp"
#include "linear-algebra/values/functions/cosh.hpp"
#include "internal/periodic_utils.hpp"

namespace OpenKalman::value
{
  /**
   * \brief Constexpr alternative to the std::cos function.
   */
#ifdef __cpp_concepts
template<value::value Arg>
  constexpr value::value auto cos(const Arg& arg)
#else
  template<typename Arg>
  constexpr auto cos(const Arg& arg, std::enable_if_t<value::value<Arg>, int> = 0)
#endif
  {
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::cos(a); } };
      return value::operation {Op{}, arg};
    }
    else
    {
      using std::cos;
      using Return = decltype(cos(arg));
      struct Op { constexpr auto operator()(const Arg& arg) { return cos(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return cos(arg);
      else if constexpr (value::complex<Arg>)
      {
        auto re = value::real(value::real(arg));
        auto im = value::real(value::imag(arg));
        if (value::isnan(re) or value::isnan(im)) return value::internal::NaN<Return>();
        using R = std::decay_t<decltype(re)>;
        if (value::isinf(re) or value::isnan(re)) return value::internal::NaN<Return>();
        auto theta = internal::scale_periodic_function(re);
        auto sinre = internal::sin_cos_impl(4, theta, theta, theta * theta * theta / -6);
        auto cosre = internal::sin_cos_impl(3, theta, R{1}, static_cast<R>(-0.5) * theta * theta);
        return value::internal::make_complex_number<Return>(cosre * value::cosh(im), -sinre * value::sinh(im));
      }
      else
      {
        if (value::isinf(arg) or value::isnan(arg)) return value::internal::NaN<Return>();
        auto theta{internal::scale_periodic_function(static_cast<Return>(arg))};
        return internal::sin_cos_impl(3, theta, Return{1}, static_cast<Return>(-0.5) * theta * theta);
      }
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_COS_HPP
