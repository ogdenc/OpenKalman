/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
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

#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/values/functions/to_number.hpp"
#include "linear-algebra/values/traits/number_type_of_t.hpp"
#include "linear-algebra/values/classes/Fixed.hpp"

namespace OpenKalman::value
{
  namespace detail
  {
    template<typename T, typename F>
    struct CastTo
    {
      static constexpr T value {static_cast<T>(value::to_number(F{}))};
      constexpr auto operator()() const { return value; }
    };
  } // namespace detail


  /**
   * \internal
   * \brief Cast a \ref value::value to another \ref value::value based on a given \ref value::number type.
   * \tparam T The \ref value::number type associated with the result
   * \tparam Arg A \ref value::value
   */
#ifdef __cpp_concepts
  template<value::number T, value::value Arg>
  constexpr value::value auto
#else
  template<typename T, typename Arg, std::enable_if_t<value::number<T> and value::value<Arg>, int> = 0>
  constexpr auto
#endif
  cast_to(Arg&& arg)
  {
    if constexpr (std::is_same_v<value::number_type_of_t<Arg>, T>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (value::fixed<Arg>)
    {
      if constexpr (value::integral<Arg>)
        return value::Fixed<T, value::to_number(std::decay_t<Arg>{})>{};
      else
        return value::Fixed<detail::CastTo<T, std::decay_t<Arg>>>{};
    }
    else
    {
      return static_cast<T>(value::to_number(std::forward<Arg>(arg)));
    }
  }

} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUES_CAST_TO_HPP
