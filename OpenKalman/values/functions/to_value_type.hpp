/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref values::to_value_type.
 */

#ifndef OPENKALMAN_VALUES_TO_VALUE_TYPE_HPP
#define OPENKALMAN_VALUES_TO_VALUE_TYPE_HPP

#include "values/concepts/fixed.hpp"

namespace OpenKalman::values
{
  /**
   * \brief Convert, if necessary, a \ref fixed or \ref dynamic value to its underlying base type.
   */
  template<typename Arg>
  constexpr decltype(auto)
  to_value_type(Arg&& arg)
  {
#ifdef __cpp_concepts
    if constexpr (requires { std::decay_t<Arg>::value; })
      return std::decay_t<Arg>::value;
    else if constexpr (requires { std::forward<Arg>(arg)(); })
      return std::forward<Arg>(arg)();
    else
      return std::forward<Arg>(arg);
#else
    if constexpr (internal::has_value_member<std::decay_t<Arg>>::value)
      return std::decay_t<Arg>::value;
    else if constexpr (internal::call_result_is_defined<std::decay_t<Arg>>::value)
      return std::forward<Arg>(arg)();
    else
      return std::forward<Arg>(arg);
#endif
  }

}

#endif
