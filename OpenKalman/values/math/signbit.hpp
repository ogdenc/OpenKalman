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
 * \brief Definition for \ref value::signbit.
 */

#ifndef OPENKALMAN_VALUE_SIGNBIT_HPP
#define OPENKALMAN_VALUE_SIGNBIT_HPP

#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/classes/operation.hpp"
#include "values/functions/internal/constexpr_callable.hpp"

namespace OpenKalman::value
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
  template<value::value Arg> requires (not value::complex<value::number_type_of_t<Arg>>)
#else
  template <typename Arg, std::enable_if_t<value::value<Arg> and not value::complex<value::number_type_of_t<Arg>>, int> = 0>
#endif
  constexpr bool signbit(const Arg& arg)
  {
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::signbit(a); } };
      return value::operation {Op{}, arg};
    }
    else
    {
      using std::signbit;
      struct Op { auto operator()(const Arg& arg) { return signbit(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return signbit(arg);
      if constexpr (value::integral<Arg>) return arg < 0;
      // Note: The result will be inaccurate if, at this stage, value::isnan(arg) == true or
      // (std::numeric_limits<Arg>::is_iec559 and arg == 0).
      return arg < 0;
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_SIGNBIT_HPP
