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
 * \file
 * \brief Definition for values::imag.
 */

#ifndef OPENKALMAN_VALUE_IMAG_HPP
#define OPENKALMAN_VALUE_IMAG_HPP

#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/functions/operation.hpp"

namespace OpenKalman::values
{
  /**
   * \brief A constexpr function to obtain the imaginary part of a (complex) number.
   * \details If arg is \ref values::complex "complex", arg must either match
   * <code>std::imag(values::to_number(arg))</code> or some defined function <code>imag(values::to_number(arg))</code>.
   * If arg is not \ref values::complex "complex" and no <code>imag</code> function is defined, the result will be
   * - <code>static_cast<double>(0)</code> if Arg is \ref values::integral "integral" or
   * - <code>static_cast<std::decay_t<Arg>>(0)</code> otherwise.
   */
#ifdef __cpp_concepts
  template<value Arg>
  constexpr value auto
#else
  template<typename Arg, std::enable_if_t<value<Arg>, int> = 0>
  constexpr auto
#endif
  imag(const Arg& arg)
  {
    if constexpr (fixed<Arg>)
    {
      struct Op { constexpr auto operator()(const number_type_of_t<Arg>& a) const { return values::imag(a); } };
      return values::operation(Op{}, arg);
    }
    else
    {
      return interface::number_traits<std::decay_t<Arg>>::imag(std::move(arg));
    }
  }


} // namespace OpenKalman::values


#endif //OPENKALMAN_VALUE_IMAG_HPP
