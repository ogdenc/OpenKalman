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
 * \brief Definition for \ref values::conj.
 */

#ifndef OPENKALMAN_VALUES_CONJ_HPP
#define OPENKALMAN_VALUES_CONJ_HPP

#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/traits/value_type_of.hpp"
#include "values/functions/operation.hpp"
#include "real.hpp"
#include "imag.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/functions/internal/constexpr_callable.hpp"


namespace OpenKalman::values
{
  /**
   * \brief A constexpr function for the complex conjugate of a (complex) number.
   */
#ifdef __cpp_concepts
  template<value Arg>
  constexpr value auto conj(const Arg& arg)
#else
  template <typename Arg, std::enable_if_t<value<Arg>, int> = 0>
  constexpr auto conj(const Arg& arg)
#endif
  {
    if constexpr (fixed<Arg>)
    {
      struct Op { constexpr auto operator()(const value_type_of_t<Arg>& a) const { return values::conj(a); } };
      return values::operation(Op{}, arg);
    }
    else
    {
      using std::conj;
      using Return = std::decay_t<decltype(conj(arg))>;
      struct Op { auto operator()(const Arg& arg) { return conj(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return conj(arg);
      else return values::internal::make_complex_number<Return>(values::real(arg), -values::imag(arg));
    }
  }


}


#endif
