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
 * \brief Definition for \ref values::asinh.
 */

#ifndef OPENKALMAN_VALUES_ASINH_HPP
#define OPENKALMAN_VALUES_ASINH_HPP

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
#include "values/math/internal/infinity.hpp"
#include "values/math/isnan.hpp"
#include "values/math/sqrt.hpp"
#include "values/math/log.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Constexpr alternative to the std::asinh function.
   */
#ifdef __cpp_concepts
  template<value Arg>
  constexpr value auto asinh(const Arg& arg)
#else
  template<typename Arg, std::enable_if_t<value<Arg>, int> = 0>
  constexpr auto asinh(const Arg& arg)
#endif
  {
    if constexpr (fixed<Arg>)
    {
      struct Op { constexpr auto operator()(const value_type_of_t<Arg>& a) const { return values::asinh(a); } };
      return values::operation(Op{}, arg);
    }
    else
    {
      using std::asinh;
      using Return = decltype(asinh(arg));
      struct Op { auto operator()(const Arg& arg) { return asinh(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return asinh(arg);
      else if constexpr (values::complex<Return>)
      {
        auto re = values::real(values::real(arg));
        auto im = values::real(values::imag(arg));
        using R = std::decay_t<decltype(re)>;
        auto sqt = values::sqrt(values::internal::make_complex_number<>(re*re - im*im + 1, 2*re*im));
        auto sqtr = values::real(sqt);
        auto sqti = values::imag(sqt);
        return values::internal::make_complex_number<Return>(values::log(values::internal::make_complex_number<R>(re + sqtr, im + sqti)));
      }
      else
      {
        if (values::isnan(arg)) return values::internal::NaN<Return>();
        if (arg == 0) return static_cast<Return>(arg);
        if constexpr (std::numeric_limits<Arg>::has_infinity)
        {
          if (arg == std::numeric_limits<Arg>::infinity()) return internal::infinity<Return>();
          if (arg == -std::numeric_limits<Arg>::infinity()) return -internal::infinity<Return>();
        }
        return values::log(arg + values::sqrt(arg * arg + 1));
      }
    }
  }


}


#endif
