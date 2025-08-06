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
 * \brief Definition for \ref values::abs.
 */

#ifndef OPENKALMAN_VALUE_ABS_HPP
#define OPENKALMAN_VALUE_ABS_HPP

#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of.hpp"
#include "values/functions/operation.hpp"
#include "values/math/real.hpp"
#include "values/math/imag.hpp"
#include "values/functions/internal/constexpr_callable.hpp"
#include "values/math/signbit.hpp"
#include "values/math/hypot.hpp"

namespace OpenKalman::values
{
  /**
   * \brief A constexpr alternative to std::abs.
   */
#ifdef __cpp_concepts
  template<value Arg>
  constexpr value auto abs(const Arg& arg)
#else
  template <typename Arg, std::enable_if_t<value<Arg>, int> = 0>
  constexpr auto abs(const Arg& arg)
#endif
  {
    if constexpr (fixed<Arg>)
    {
      struct Op { constexpr auto operator()(const number_type_of_t<Arg>& a) const { return values::abs(a); } };
      return values::operation(Op{}, arg);
    }
    else
    {
      using std::abs;
      using Return = std::decay_t<decltype(abs(arg))>;
      struct Op { auto operator()(const Arg& arg) { return abs(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return abs(arg);
      else if constexpr (complex<Arg>)
        return static_cast<Return>(values::hypot(values::real(arg), values::imag(arg)));
      else
        return static_cast<Return>(values::signbit(arg) ? -arg : arg);
    }
  }


} // namespace OpenKalman::values


#endif //OPENKALMAN_VALUE_ABS_HPP
