/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref get_scalar_constant_value.
 */

#ifndef OPENKALMAN_GET_SCALAR_CONSTANT_VALUE_HPP
#define OPENKALMAN_GET_SCALAR_CONSTANT_VALUE_HPP


namespace OpenKalman
{
  /**
   * \brief Get the \ref scalar value of a \ref scalar_constant
   */
#ifdef __cpp_concepts
  template<scalar_constant Arg>
  constexpr scalar_type decltype(auto) get_scalar_constant_value(Arg&& arg)
#else
  template<typename Arg, std::enable_if_t<scalar_constant<Arg>, int> = 0>
  constexpr decltype(auto) get_scalar_constant_value(Arg&& arg)
#endif
  {
    using T = std::decay_t<Arg>;
#ifdef __cpp_concepts
    if constexpr (requires { {T::value} -> scalar_type; }) return T::value;
    else if constexpr (requires { {arg()} -> scalar_type; }) return std::forward<Arg>(arg)();
    else return std::forward<Arg>(arg);
#else
    if constexpr (internal::has_value_member<T>::value) return T::value;
    else if constexpr (internal::call_result_is_scalar<T>::value or internal::is_runtime_scalar<T>::value)
      return std::forward<Arg>(arg)();
    else { static_assert(scalar_type<Arg>); return std::forward<Arg>(arg); }
#endif
  }


} // namespace OpenKalman

#endif //OPENKALMAN_GET_SCALAR_CONSTANT_VALUE_HPP
