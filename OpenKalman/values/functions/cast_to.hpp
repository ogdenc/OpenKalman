/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref value::cast_to.
 */

#ifndef OPENKALMAN_VALUES_CAST_TO_HPP
#define OPENKALMAN_VALUES_CAST_TO_HPP

#include "values/concepts/number.hpp"
#include "values/functions/to_number.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/classes/Fixed.hpp"

namespace OpenKalman::value
{
#if __cpp_nontype_template_args < 201911L
  namespace detail
  {
    template<typename Arg, typename T>
    struct FixedCast
    {
      using value_type = T;
      static constexpr auto value {static_cast<value_type>(value::fixed_number_of_v<Arg>)};
      using type = FixedCast;
      constexpr operator value_type() const { return value; }
      constexpr value_type operator()() const { return value; }
    };
  } // namespace detail
#endif


  /**
   * \internal
   * \brief Cast a \ref value::value to another \ref value::value based on a given \ref value::number type.
   * \tparam T The \ref value::number type associated with the result
   * \tparam Arg A \ref value::value
   */
#ifdef __cpp_concepts
  template<value::number T, value::value Arg>
  constexpr value::value decltype(auto)
#else
  template<typename T, typename Arg, std::enable_if_t<value::number<T> and value::value<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  cast_to(Arg&& arg)
  {
    if constexpr (std::is_same_v<value::number_type_of_t<Arg>, T>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (value::fixed<Arg>)
    {
      constexpr auto x = value::fixed_number_of_v<Arg>;
#if __cpp_nontype_template_args >= 201911L
      return value::Fixed<T, x>{};
#else
      if constexpr (x == static_cast<std::intmax_t>(x))
      {
        return value::Fixed<T, static_cast<std::intmax_t>(x)>{};
      }
      else
      {
        return detail::FixedCast<std::decay_t<Arg>, T>{};
      }
#endif
    }
    else
    {
      return static_cast<T>(value::to_number(std::forward<Arg>(arg)));
    }
  }

} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUES_CAST_TO_HPP
