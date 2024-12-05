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
 * \brief Definition for \ref value::log.
 */

#ifndef OPENKALMAN_VALUE_LOG_HPP
#define OPENKALMAN_VALUE_LOG_HPP

#include <limits>
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
#include "linear-algebra/values/functions/internal/infinity.hpp"
#include "linear-algebra/values/functions/isnan.hpp"
#include "linear-algebra/values/functions/signbit.hpp"
#include "linear-algebra/values/functions/copysign.hpp"
#include "internal/math_utils.hpp"
#include "internal/log_utils.hpp"
#include "internal/atan_utils.hpp"

namespace OpenKalman::value
{
  /**
   * \brief Constexpr alternative to the std::log function.
   */
#ifdef __cpp_concepts
  template<value::value Arg>
  constexpr value::value auto log(const Arg& arg)
#else
  template<typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
  constexpr auto log(const Arg& arg)
#endif
  {
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::log(a); } };
      return value::operation {Op{}, arg};
    }
    else
    {
      using std::log;
      using Return = decltype(log(arg));
      struct Op { constexpr auto operator()(const Arg& arg) { return log(arg); } };
      if (value::internal::constexpr_callable<Op>(arg)) return log(arg);
      else if constexpr (value::complex<Return>)
      {
        auto re = value::real(value::real(arg));
        auto im = value::real(value::imag(arg));
        using R = decltype(re);
        auto a = static_cast<R>(0.5) * value::log(re * re + im * im);
        if constexpr (not std::numeric_limits<value::real_type_of_t<Arg>>::is_iec559) if (value::imag(arg) == 0)
          return value::internal::make_complex_number<Return>(a,
            value::copysign(value::signbit(value::real(arg)) ? numbers::pi_v<R> : 0, value::imag(arg)));
        return value::internal::make_complex_number<Return>(a, internal::atan2_impl(im, re));
      }
      else
      {
        if (value::isnan(arg)) return value::internal::NaN<Return>();
        if constexpr (std::numeric_limits<Arg>::has_infinity)
          if (arg == std::numeric_limits<Arg>::infinity()) return std::numeric_limits<Return>::infinity();
        if (arg == 1) return static_cast<Return>(+0.);
        else if (arg == 0) return -value::internal::infinity<Return>();
        else if (arg < 0) return value::internal::NaN<Return>();
        auto [scaled, corr] = arg >= 16 ? internal::log_scaling_gt(value::real(arg)) : internal::log_scaling_lt(value::real(arg));
        return internal::log_impl(scaled) + corr;
      }
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_LOG_HPP
