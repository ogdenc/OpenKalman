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
 * \file
 * \brief Definition for values::real.
 */

#ifndef OPENKALMAN_VALUES_REAL_HPP
#define OPENKALMAN_VALUES_REAL_HPP

#include "values/concepts/fixed.hpp"
#include "values/concepts/value.hpp"
#include "values/functions/operation.hpp"

namespace OpenKalman::values
{
  /**
   * \brief A constexpr function to obtain the real part of a (complex) number.
   * \details If arg is \ref values::complex "complex", arg must either match
   * <code>std::real(values::to_value_type(arg))</code> or some defined function <code>real(values::to_value_type(arg))</code>.
   * If arg is not \ref values::complex "complex" and no <code>real</code> function is defined, the result will be
   * - <code>static_cast<double>(std::forward<Arg>(arg))</code> if Arg is \ref values::integral "integral" or
   * - <code>static_cast<std::decay_t<Arg>>(std::forward<Arg>(arg))</code> otherwise.
   */
#ifdef __cpp_concepts
  template<value Arg>
  constexpr value auto
#else
  template<typename Arg, std::enable_if_t<value<Arg>, int> = 0>
  constexpr auto
#endif
  real(const Arg& arg)
  {
    if constexpr (fixed<Arg>)
    {
      struct Op { constexpr auto operator()(const value_type_of_t<Arg>& a) const { return values::real(a); } };
      return values::operation(Op{}, arg);
    }
    else
    {
      return interface::number_traits<std::decay_t<Arg>>::real(std::move(arg));
    }
  }


}


#endif
