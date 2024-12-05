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
 * \brief Definition for \ref value::asinh.
 */

#ifndef OPENKALMAN_VALUE_ASINH_HPP
#define OPENKALMAN_VALUE_ASINH_HPP

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
#include "linear-algebra/values/functions/internal/infinity.hpp"
#include "linear-algebra/values/functions/isnan.hpp"
#include "linear-algebra/values/functions/sqrt.hpp"
#include "linear-algebra/values/functions/log.hpp"

namespace OpenKalman::value
{
  /**
   * \brief Constexpr alternative to the std::asinh function.
   */
#ifdef __cpp_concepts
  template<value::value Arg>
  constexpr value::value auto asinh(const Arg& arg)
#else
  template<typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
  constexpr auto asinh(const Arg& arg)
#endif
  {
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::asinh(a); } };
      return value::operation {Op{}, arg};
    }
    else
    {
      using std::asinh;
      using Return = decltype(asinh(arg));
      struct Op { constexpr auto operator()(const Arg& arg) { return asinh(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return asinh(arg);
      else if constexpr (value::complex<Return>)
      {
        auto re = value::real(value::real(arg));
        auto im = value::real(value::imag(arg));
        using R = std::decay_t<decltype(re)>;
        auto sqt = value::sqrt(value::internal::make_complex_number(re*re - im*im + 1, 2*re*im));
        auto sqtr = value::real(sqt);
        auto sqti = value::imag(sqt);
        return value::internal::make_complex_number<Return>(value::log(value::internal::make_complex_number<R>(re + sqtr, im + sqti)));
      }
      else
      {
        if (value::isnan(arg)) return value::internal::NaN<Return>();
        if (arg == 0) return static_cast<Return>(arg);
        if constexpr (std::numeric_limits<Arg>::has_infinity)
        {
          if (arg == std::numeric_limits<Arg>::infinity()) return internal::infinity<Return>();
          if (arg == -std::numeric_limits<Arg>::infinity()) return -internal::infinity<Return>();
        }
        return value::log(arg + value::sqrt(arg * arg + 1));
      }
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_ASINH_HPP
