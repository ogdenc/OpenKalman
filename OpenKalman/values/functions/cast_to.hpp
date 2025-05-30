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
 * \brief Definition for \ref values::cast_to.
 */

#ifndef OPENKALMAN_VALUES_CAST_TO_HPP
#define OPENKALMAN_VALUES_CAST_TO_HPP

#include "values/concepts/number.hpp"
#include "values/functions/to_number.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/classes/Fixed.hpp"

namespace OpenKalman::values
{
#if __cpp_nontype_template_args < 201911L
  namespace detail
  {
    template<typename Arg, typename T>
    struct FixedCast
    {
      using value_type = T;
      static constexpr auto value {static_cast<value_type>(values::fixed_number_of_v<Arg>)};
      using type = FixedCast;
      constexpr operator value_type() const { return value; }
      constexpr value_type operator()() const { return value; }
    };
  } // namespace detail
#endif


  /**
   * \internal
   * \brief Cast a \ref values::value to another \ref values::value based on a given \ref values::number type.
   * \tparam T The \ref values::number type associated with the result
   * \tparam Arg A \ref values::value
   */
#ifdef __cpp_concepts
  template<values::number T, values::value Arg>
  constexpr values::value decltype(auto)
#else
  template<typename T, typename Arg, std::enable_if_t<values::number<T> and values::value<Arg>, int> = 0>
  constexpr decltype(auto)
#endif
  cast_to(Arg&& arg)
  {
    if constexpr (std::is_same_v<values::number_type_of_t<Arg>, T>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (values::fixed<Arg>)
    {
      constexpr auto x = values::fixed_number_of_v<Arg>;
#if __cpp_nontype_template_args >= 201911L
      return values::Fixed<T, x>{};
#else
      if constexpr (x == static_cast<std::intmax_t>(x))
      {
        return values::Fixed<T, static_cast<std::intmax_t>(x)>{};
      }
      else
      {
        return detail::FixedCast<std::decay_t<Arg>, T>{};
      }
#endif
    }
    else
    {
      return static_cast<T>(values::to_number(std::forward<Arg>(arg)));
    }
  }

} // namespace OpenKalman::values

#endif //OPENKALMAN_VALUES_CAST_TO_HPP
