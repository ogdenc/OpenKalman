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
 * \brief Definition for \ref values::atan2.
 */

#ifndef OPENKALMAN_VALUE_ATAN2_HPP
#define OPENKALMAN_VALUE_ATAN2_HPP

#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/traits/real_type_of_t.hpp"
#include "values/classes/operation.hpp"
#include "values/math/real.hpp"
#include "values/math/imag.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "values/math/internal/NaN.hpp"
#include "values/math/isnan.hpp"
#include "internal/math_utils.hpp"
#include "internal/atan_utils.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Constexpr alternative to the std::atan2 function.
   * \details Unlike the standard function, this one accepts complex arguments.
   */
#ifdef __cpp_concepts
  template <values::value Y, values::value X> requires
    std::common_with<values::number_type_of_t<Y>, values::number_type_of_t<X>>
  constexpr values::value auto
#else
  template <typename Y, typename X, std::enable_if_t<values::value<Y> and values::value<X>, int> = 0>
  constexpr auto
#endif
  atan2(const Y& y_arg, const X& x_arg)
  {
    if constexpr (not values::number<Y> or not values::number<X>)
    {
      struct Op
      {
        using N = std::common_type_t<values::number_type_of_t<Y>, values::number_type_of_t<X>>;
        constexpr auto operator()(const N& y, const N& x) const { return values::atan2(y, x); }
      };
      return values::operation {Op{}, y_arg, x_arg};
    }
    else
    {
      using Arg = std::common_type_t<Y, X>;
      if constexpr (values::complex<Arg>)
      {
        using Return = std::decay_t<Arg>;
        using R = real_type_of_t<real_type_of_t<Arg>>;
        auto yr = values::real(values::real(y_arg));
        auto yi = values::real(values::imag(y_arg));
        auto xr = values::real(values::real(x_arg));
        auto xi = values::real(values::imag(x_arg));
        auto pi = numbers::pi_v<R>;
        auto halfpi = static_cast<R>(0.5) * pi;
        if (xr == 0 and xi == 0)
        {
          if (yr > 0) return values::internal::make_complex_number<Return>(halfpi, 0);
          else if (yr < 0) return values::internal::make_complex_number<Return>(-halfpi, 0);
          else return values::internal::make_complex_number<Return>(0, 0);
        }
        if (yr == 0 and yi == 0)
        {
          if (xr < 0) return values::internal::make_complex_number<Return>(pi, 0);
          else return values::internal::make_complex_number<Return>(0, 0);
        }
        else
        {
          auto denom = xr*xr + xi*xi;
          auto raw = internal::atan_impl_general(values::internal::make_complex_number((yr * xr + yi * xi) / denom, (yi * xr - yr * xi) / denom));
          auto raw_r = values::real(raw);
          auto raw_i = values::imag(raw);
          if (raw_r > pi) return values::internal::make_complex_number<Return>(raw_r - pi, raw_i);
          else if (raw_r < -pi) return values::internal::make_complex_number<Return>(raw_r + pi, raw_i);
          else return values::internal::make_complex_number<Return>(raw_r, raw_i);
        }
      }
      else
      {
        using std::atan2;
        using Return = decltype(atan2(y_arg, x_arg));
        struct Op { auto operator()(const Arg& y_arg, const Arg& x_arg) { return atan2(y_arg, x_arg); } };
        if (values::internal::constexpr_callable<Op>(y_arg, x_arg)) return atan2(y_arg, x_arg);
        if (values::isnan(y_arg) or values::isnan(x_arg)) return values::internal::NaN<Return>();
        return internal::atan2_impl(static_cast<Return>(y_arg), static_cast<Return>(x_arg));
      }
    }
  }


} // namespace OpenKalman::values


#endif //OPENKALMAN_VALUE_ATAN2_HPP
