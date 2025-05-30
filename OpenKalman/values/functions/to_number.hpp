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
 * \brief Definition for \ref values::to_number.
 */

#ifndef OPENKALMAN_VALUE_TO_NUMBER_HPP
#define OPENKALMAN_VALUE_TO_NUMBER_HPP

#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Convert any \ref values::value to a \ref values::number
   */
#ifdef __cpp_concepts
  template<value Arg>
  constexpr number auto
  to_number(Arg arg)
#else
  template<typename Arg>
  constexpr auto
  to_number(Arg arg)
#endif
  {
#ifdef __cpp_concepts
    if constexpr (requires { {std::decay_t<Arg>::value} -> number; }) return std::decay_t<Arg>::value;
    else if constexpr (requires { {std::move(arg)()} -> number; }) return std::move(arg)();
    else return std::move(arg);
#else
    static_assert(value<Arg>);
    if constexpr (internal::has_value_member<std::decay_t<Arg>>::value) return std::decay_t<Arg>::value;
    else if constexpr (internal::call_result_is_fixed<std::decay_t<Arg>>::value or internal::is_dynamic<std::decay_t<Arg>>::value)
      return std::move(arg)();
    else { static_assert(number<Arg>); return std::move(arg); }
#endif
  }


} // namespace OpenKalman::values

#endif //OPENKALMAN_VALUE_TO_NUMBER_HPP
