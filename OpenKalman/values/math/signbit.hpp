/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \brief Definition for \ref values::signbit.
 */

#ifndef OPENKALMAN_VALUES_SIGNBIT_HPP
#define OPENKALMAN_VALUES_SIGNBIT_HPP

#include "values/concepts/value.hpp"
#include "values/concepts/complex.hpp"
#include "values/traits/value_type_of.hpp"
#include "values/functions/operation.hpp"
#include "values/functions/internal/constexpr_callable.hpp"

namespace OpenKalman::values
{
  /**
   * \brief A constexpr function analogous to std::signbit.
   * \details If the compiler offers a constexpr version of std::signbit, it will be called.
   * Otherwise, the argument will be compared with zero.
   * \note In most pre-c++23 compilations, this function will be inaccurate if the argument is either -NaN or -0.0.
   * This is because prior to the c++23 standard library, there was no way to determine,
   * at compile time, the sign of either ±NaN or ±0.0.
   */
#ifdef __cpp_concepts
  template<value Arg> requires (not complex<value_type_of_t<Arg>>)
  constexpr std::convertible_to<bool> auto
#else
  template <typename Arg, std::enable_if_t<value<Arg> and not complex<value_type_of_t<Arg>>, int> = 0>
  constexpr auto
#endif
  signbit(const Arg& arg)
  {
    if constexpr (fixed<Arg>)
    {
      struct Op { constexpr auto operator()(const value_type_of_t<Arg>& a) const { return values::signbit(a); } };
      return values::operation(Op{}, arg);
    }
    else
    {
      using std::signbit;
      struct Op { auto operator()(const Arg& arg) { return signbit(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return signbit(arg);
      // Note: The result will be inaccurate if, at this stage, values::isnan(arg) == true or
      // (std::numeric_limits<Arg>::is_iec559 and arg == 0).
      return arg < 0;
    }
  }


}


#endif
