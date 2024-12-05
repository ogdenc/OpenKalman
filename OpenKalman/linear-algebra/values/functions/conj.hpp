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
 * \brief Definition for \ref value::conj.
 */

#ifndef OPENKALMAN_VALUE_CONJ_HPP
#define OPENKALMAN_VALUE_CONJ_HPP

#include "linear-algebra/values/interface/number_traits.hpp"
#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/values/concepts/value.hpp"
#include "linear-algebra/values/traits/number_type_of_t.hpp"
#include "linear-algebra/values/classes/operation.hpp"
#include "real.hpp"
#include "imag.hpp"
#include "internal/make_complex_number.hpp"
#include "internal/constexpr_callable.hpp"


namespace OpenKalman::value
{
  /**
   * \brief A constexpr function for the complex conjugate of a (complex) number.
   */
#ifdef __cpp_concepts
  template<value::value Arg>
  constexpr value::value auto conj(const Arg& arg)
#else
  template <typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
  constexpr auto conj(const Arg& arg)
#endif
  {
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::conj(a); } };
      return value::operation {Op{}, arg};
    }
    else
    {
      using std::conj;
      using Return = std::decay_t<decltype(conj(arg))>;
      struct Op { constexpr auto operator()(const Arg& arg) { return conj(arg); } };
      if (internal::constexpr_callable<Op>(arg)) return conj(arg);
      else return value::internal::make_complex_number<Return>(value::real(arg), -value::imag(arg));
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_CONJ_HPP
