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
 * \brief Definition for \ref value::isinf.
 */

#ifndef OPENKALMAN_VALUE_ISINF_HPP
#define OPENKALMAN_VALUE_ISINF_HPP

#include <limits>
#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/values/functions/internal/constexpr_callable.hpp"

namespace OpenKalman::value
{
  /**
   * \brief Constexpr alternative to std::isinf. Checks whether the input is positive or negative infinity.
   */
#ifdef __cpp_concepts
  template <value::value Arg> requires (not value::complex<value::number_type_of_t<Arg>>)
#else
  template <typename Arg, std::enable_if_t<value::value<Arg> and (not value::complex<value::number_type_of_t<Arg>>), int> = 0>
#endif
  constexpr bool isinf(const Arg& arg)
  {
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::isinf(a); } };
      return value::operation {Op{}, arg};
    }
    else
    {
      using std::isinf;
      struct Op { constexpr auto operator()(const Arg& arg) { return isinf(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return isinf(arg);
      if constexpr (std::numeric_limits<Arg>::has_infinity)
        return arg == std::numeric_limits<Arg>::infinity() or arg == -std::numeric_limits<Arg>::infinity();
      return false;
    }
  }

} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_ISINF_HPP
