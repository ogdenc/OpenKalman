/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \brief Definition for \ref values::isnan.
 */

#ifndef OPENKALMAN_VALUES_ISNAN_HPP
#define OPENKALMAN_VALUES_ISNAN_HPP

#include <limits>
#include "values/concepts/number.hpp"
#include "values/functions/internal/constexpr_callable.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Constexpr alternative to std::isnan. Checks whether the input is not a number (NaN).
   */
#ifdef __cpp_concepts
  template <value Arg> requires (not values::complex<values::value_type_of_t<Arg>>)
#else
  template <typename Arg, std::enable_if_t<value<Arg> and (not values::complex<values::value_type_of_t<Arg>>), int> = 0>
#endif
  constexpr bool isnan(const Arg& arg)
  {
    if constexpr (fixed<Arg>)
    {
      struct Op { constexpr auto operator()(const value_type_of_t<Arg>& a) const { return values::isnan(a); } };
      return values::operation(Op{}, arg);
    }
    else
    {
      using std::isnan;
      struct Op { auto operator()(const Arg& arg) { return isnan(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return isnan(arg);
      return arg != arg;
    }
  }

}


#endif
