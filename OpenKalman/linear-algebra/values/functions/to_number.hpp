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

#ifndef OPENKALMAN_GET_SCALAR_CONSTANT_VALUE_HPP
#define OPENKALMAN_GET_SCALAR_CONSTANT_VALUE_HPP

#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/values/concepts/scalar.hpp"

namespace OpenKalman::value
{
  /**
   * \brief Convert any \ref value::scalar to a \ref value::number
   */
#ifdef __cpp_concepts
  template<value::scalar Arg>
  constexpr value::number decltype(auto) to_number(Arg&& arg)
#else
  template<typename Arg, std::enable_if_t<value::scalar<Arg>, int> = 0>
  constexpr decltype(auto) to_number(Arg&& arg)
#endif
  {
    using T = std::decay_t<Arg>;
#ifdef __cpp_concepts
    if constexpr (requires { {T::value} -> value::number; }) return T::value;
    else if constexpr (requires { {arg()} -> value::number; }) return std::forward<Arg>(arg)();
    else return std::forward<Arg>(arg);
#else
    if constexpr (value::internal::has_value_member<T>::value) return T::value;
    else if constexpr (value::internal::call_result_is_scalar<T>::value or value::internal::is_runtime_scalar<T>::value)
      return std::forward<Arg>(arg)();
    else { static_assert(value::number<Arg>); return std::forward<Arg>(arg); }
#endif
  }


} // namespace OpenKalman::value

#endif //OPENKALMAN_GET_SCALAR_CONSTANT_VALUE_HPP
