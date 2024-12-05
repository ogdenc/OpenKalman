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
 * \brief Definition for \ref value::atan2.
 */

#ifndef OPENKALMAN_VALUE_ATAN2_HPP
#define OPENKALMAN_VALUE_ATAN2_HPP

#include "linear-algebra/values/interface/number_traits.hpp"
#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/values/concepts/value.hpp"
#include "linear-algebra/values/traits/number_type_of_t.hpp"
#include "linear-algebra/values/traits/real_type_of_t.hpp"
#include "linear-algebra/values/classes/operation.hpp"
#include "linear-algebra/values/functions/real.hpp"
#include "linear-algebra/values/functions/imag.hpp"
#include "linear-algebra/values/functions/internal/make_complex_number.hpp"
#include "linear-algebra/values/functions/internal/constexpr_callable.hpp"
#include "linear-algebra/values/functions/internal/NaN.hpp"
#include "linear-algebra/values/functions/isnan.hpp"
#include "internal/math_utils.hpp"
#include "internal/atan_utils.hpp"

namespace OpenKalman::value
{
  /**
   * \brief Constexpr alternative to the std::atan2 function.
   * \details Unlike the standard function, this one accepts complex arguments.
   */
#ifdef __cpp_concepts
  template <value::value Y, value::value X> requires
    std::common_with<value::number_type_of_t<Y>, value::number_type_of_t<X>>
  constexpr value::value auto
#else
  template <typename Y, typename X, std::enable_if_t<value::value<Y> and value::value<X>, int> = 0>
  constexpr auto
#endif
  atan2(const Y& y_arg, const X& x_arg)
  {
    if constexpr (not value::number<Y> or not value::number<X>)
    {
      struct Op
      {
        using N = std::common_type_t<value::number_type_of_t<Y>, value::number_type_of_t<X>>;
        constexpr auto operator()(const N& y, const N& x) const { return value::atan2(y, x); }
      };
      return value::operation {Op{}, y_arg, x_arg};
    }
    else
    {
      using Arg = std::common_type_t<Y, X>;
      if constexpr (value::complex<Arg>)
      {
        using Return = std::decay_t<Arg>;
        using R = real_type_of_t<real_type_of_t<Arg>>;
        auto yr = value::real(value::real(y_arg));
        auto yi = value::real(value::imag(y_arg));
        auto xr = value::real(value::real(x_arg));
        auto xi = value::real(value::imag(x_arg));
        auto pi = numbers::pi_v<R>;
        auto halfpi = static_cast<R>(0.5) * pi;
        if (xr == 0 and xi == 0)
        {
          if (yr > 0) return value::internal::make_complex_number<Return>(halfpi, 0);
          else if (yr < 0) return value::internal::make_complex_number<Return>(-halfpi, 0);
          else return value::internal::make_complex_number<Return>(0, 0);
        }
        if (yr == 0 and yi == 0)
        {
          if (xr < 0) return value::internal::make_complex_number<Return>(pi, 0);
          else return value::internal::make_complex_number<Return>(0, 0);
        }
        else
        {
          auto denom = xr*xr + xi*xi;
          auto raw = internal::atan_impl_general(value::internal::make_complex_number((yr * xr + yi * xi) / denom, (yi * xr - yr * xi) / denom));
          auto raw_r = value::real(raw);
          auto raw_i = value::imag(raw);
          if (raw_r > pi) return value::internal::make_complex_number<Return>(raw_r - pi, raw_i);
          else if (raw_r < -pi) return value::internal::make_complex_number<Return>(raw_r + pi, raw_i);
          else return value::internal::make_complex_number<Return>(raw_r, raw_i);
        }
      }
      else
      {
        using std::atan2;
        using Return = decltype(atan2(y_arg, x_arg));
        struct Op { constexpr auto operator()(const Arg& y_arg, const Arg& x_arg) { return atan2(y_arg, x_arg); } };
        if (value::internal::constexpr_callable<Op>(y_arg, x_arg)) return atan2(y_arg, x_arg);
        if (value::isnan(y_arg) or value::isnan(x_arg)) return value::internal::NaN<Return>();
        return internal::atan2_impl(static_cast<Return>(y_arg), static_cast<Return>(x_arg));
      }
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_ATAN2_HPP
