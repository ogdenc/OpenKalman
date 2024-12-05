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
 * \brief Definition for value::imag.
 */

#ifndef OPENKALMAN_VALUE_IMAG_HPP
#define OPENKALMAN_VALUE_IMAG_HPP

#include "linear-algebra/values/interface/number_traits.hpp"
#include "linear-algebra/values/interface/number_traits_defined.hpp"
#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/values/concepts/value.hpp"
#include "linear-algebra/values/concepts/integral.hpp"
#include "linear-algebra/values/traits/number_type_of_t.hpp"
#include "linear-algebra/values/classes/operation.hpp"

namespace OpenKalman::value
{
  /**
   * \brief A constexpr function to obtain the imaginary part of a (complex) number.
   */
#ifdef __cpp_concepts
  template<value::value Arg>
  constexpr value::value decltype(auto)
#else
  template<typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  imag(Arg&& arg)
  {
    if constexpr (not value::number<Arg>)
    {
      struct Op { constexpr auto operator()(const value::number_type_of_t<Arg>& a) const { return value::imag(a); } };
      return value::operation {Op{}, std::forward<Arg>(arg)};
    }
    else if constexpr (value::complex<Arg>)
    {
      if constexpr (interface::imag_defined_for<Arg>)
      {
        return interface::number_traits<std::decay_t<Arg>>::imag(std::forward<Arg>(arg));
      }
      else
      {
        using std::imag;
        return imag(std::forward<Arg>(arg));
      }
    }
    else
    {
      return static_cast<std::conditional_t<value::integral<Arg>, double, std::decay_t<Arg>>>(0);
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_IMAG_HPP
