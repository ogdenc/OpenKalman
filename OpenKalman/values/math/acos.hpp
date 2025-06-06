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
 * \brief Definition for \ref values::acos.
 */

#ifndef OPENKALMAN_VALUE_ACOS_HPP
#define OPENKALMAN_VALUE_ACOS_HPP

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
#include "values/math/asin.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Constexpr alternative to the std::acos function.
   */
#ifdef __cpp_concepts
  template<value Arg>
  constexpr value auto acos(const Arg& arg)
#else
  template <typename Arg, std::enable_if_t<value<Arg>, int> = 0>
  constexpr auto acos(const Arg& arg)
#endif
  {
    if constexpr (not number<Arg>)
    {
      struct Op { constexpr auto operator()(const number_type_of_t<Arg>& a) const { return values::acos(a); } };
      return operation {Op{}, arg};
    }
    else
    {
      using std::acos;
      using Return = std::decay_t<decltype(acos(arg))>;
      struct Op { auto operator()(const Arg& arg) { return acos(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return acos(arg);
      else if constexpr (values::complex<Return>)
      {
        using R = real_type_of_t<real_type_of_t<Arg>>;
        auto s = values::asin(internal::make_complex_number<R>(arg));
        return internal::make_complex_number<Return>(numbers::pi_v<R> / 2 - values::real(s), - values::imag(s));
      }
      else
      {
        using R = real_type_of_t<Arg>;
        if (arg == 1) return static_cast<Return>(+0.);
        auto s = values::asin(arg);
        if (values::isnan(s)) return internal::NaN<Return>();
        return numbers::pi_v<R> / 2 - s;
      }
    }
  }


} // namespace OpenKalman::values


#endif //OPENKALMAN_VALUE_ACOS_HPP
