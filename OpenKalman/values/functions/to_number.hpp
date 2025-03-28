/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref value::to_number.
 */

#ifndef OPENKALMAN_VALUE_TO_NUMBER_HPP
#define OPENKALMAN_VALUE_TO_NUMBER_HPP

#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"

namespace OpenKalman::value
{
  /**
   * \brief Convert any \ref value::value to a \ref value::number
   */
#ifdef __cpp_concepts
  template<value::value Arg>
  constexpr value::number auto
  to_number(Arg arg)
#else
  template<typename Arg, std::enable_if_t<value::value<Arg>, int> = 0>
  constexpr auto
  to_number(Arg arg)
#endif
  {
    using T = std::decay_t<Arg>;
#ifdef __cpp_concepts
    if constexpr (requires { {T::value} -> value::number; }) return T::value;
    else if constexpr (requires { {std::move(arg)()} -> value::number; }) return std::move(arg)();
    else return arg;
#else
    if constexpr (value::internal::has_value_member<T>::value) return T::value;
    else if constexpr (internal::call_result_is_fixed<T>::value or internal::is_dynamic<T>::value)
      return std::move(arg)();
    else { static_assert(value::number<Arg>); return arg; }
#endif
  }


} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUE_TO_NUMBER_HPP
