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

#include <cstdint>
#include "values/concepts/number.hpp"
#include "values/concepts/complex.hpp"
#include "values/functions/to_number.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/traits/number_type_of.hpp"
#include "values/traits/complex_type_of.hpp"
#include "values/classes/Fixed.hpp"

namespace OpenKalman::values
{
#if __cpp_nontype_template_args < 201911L
  namespace detail
  {
    template<typename T, typename Arg>
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

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_complexible : std::false_type {};

    template<typename T>
    struct is_complexible<T, std::void_t<typename complex_type_of<T>::type>> : std::true_type {};


    template<typename T, typename Arg, typename = void>
    struct is_fixable : std::false_type {};

    template<typename T, typename Arg>
    struct is_fixable<T, Arg, std::enable_if_t<(std::bool_constant<(static_cast<T>(fixed_number_of<Arg>::value) == static_cast<T>(fixed_number_of<Arg>::value))>::value)>>
      : std::true_type {};
  }
#endif


  /**
   * \internal
   * \brief Cast a \ref values::value to another \ref values::value with a particular underlying real \ref values::number type.
   * \details If the argument is complex, the result will be complex. If the argument is fixed, the result may or may not be fixed.
   * \tparam T The \ref values::number type associated with the result
   * \tparam Arg A \ref values::value
   */
#ifdef __cpp_concepts
  template<number T, value Arg> requires (not complex<T>) and std::same_as<T, std::decay_t<T>> and
    requires { static_cast<T>(real(to_number(std::declval<Arg&&>()))); } and
    (not complex<Arg> or requires { typename complex_type_of<T>::type; }) and
    (complex<Arg> or not fixed<Arg> or requires { typename Fixed<T, fixed_number_of_v<Arg>>; })
  constexpr value decltype(auto)
#else
  template<typename T, typename Arg, std::enable_if_t<
    number<T> and value<Arg> and (not complex<T>) and std::is_same_v<T, std::decay_t<T>> and
    (not complex<Arg> or detail::is_complexible<T>::value) and
    (complex<Arg> or not fixed<Arg> or detail::is_fixable<T, Arg>::value), int> = 0,
    typename = std::void_t<decltype(static_cast<T>(real(to_number(std::declval<Arg&&>()))))>>
  constexpr decltype(auto)
#endif
  cast_to(Arg&& arg)
  {
    if constexpr (std::is_same_v<T, number_type_of_t<Arg>>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (complex<Arg>)
    {
      return internal::make_complex_number<T>(std::forward<Arg>(arg));
    }
    else if constexpr (fixed<Arg>)
    {
      constexpr auto x = fixed_number_of_v<Arg>;
#if __cpp_nontype_template_args >= 201911L
      return Fixed<T, x>{};
#else
      if constexpr (x == static_cast<std::intmax_t>(x)) return Fixed<T, static_cast<std::intmax_t>(x)>{};
      else return detail::FixedCast<T, std::decay_t<Arg>>{};
#endif
    }
    else
    {
      return static_cast<T>(values::to_number(std::forward<Arg>(arg)));
    }
  }

} // namespace OpenKalman::values

#endif //OPENKALMAN_VALUES_CAST_TO_HPP
