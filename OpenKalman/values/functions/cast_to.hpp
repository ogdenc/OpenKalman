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
#include "basics/basics.hpp"
#include "values/concepts/value.hpp"
#include "values/concepts/complex.hpp"
#include "values/functions/to_value_type.hpp"
#include "values/functions/internal/make_complex_number.hpp"
#include "values/traits/value_type_of.hpp"
#include "values/classes/fixed_value.hpp"

namespace OpenKalman::values
{
  /**
   * \brief A \ref fixed value cast from another \ref fixed value
   * \tparam T The new value type
   * \tparam Arg The underlying fixed value
   */
  template<typename T, typename Arg>
  struct fixed_cast
  {
    using value_type = T;
    static constexpr auto value {static_cast<value_type>(values::fixed_value_of_v<Arg>)};
    using type = fixed_cast;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }
  };


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename Arg, typename T, typename = void>
    struct value_type_convertible : std::false_type {};

    template<typename Arg, typename T>
    struct value_type_convertible<Arg, T, std::enable_if_t<
      stdex::convertible_to<typename value_type_of<Arg>::type, T>>>
      : std::true_type {};


    template<typename T, typename Arg, typename = void, typename = void>
    struct is_fixable : std::false_type {};

    template<typename T, typename Arg>
    struct is_fixable<T, Arg, std::void_t<fixed_value<T, fixed_value_of_v<Arg>>>>
      : std::true_type {};


    template<typename Arg, typename T, typename = void>
    struct real_types_convertible : std::false_type {};

    template<typename Arg, typename T>
    struct real_types_convertible<Arg, T, std::enable_if_t<
      stdex::convertible_to<typename real_type_of<Arg>::type, typename real_type_of<T>::type> >>
      : std::true_type {};
  }
#endif


  /**
   * \internal
   * \brief Cast a value to another value having a particular underlying value type.
   * \details If the argument is \ref fixed, the result may or may not be fixed.
   * \tparam T The underlying value type associated with the result
   * \tparam Arg A \ref value
   */
#ifdef __cpp_concepts
  template<typename T, typename Arg> requires
    std::same_as<T, std::decay_t<T>> and
    (not number<T> or not complex<Arg>) and
    std::convertible_to<value_type_of_t<Arg>, T>
#else
  template<typename T, typename Arg, std::enable_if_t<
    (not number<T> or not complex<Arg>) and
    std::is_same_v<T, std::decay_t<T>> and
    detail::value_type_convertible<Arg, T>::value, int> = 0>
#endif
  constexpr decltype(auto)
  cast_to(Arg&& arg)
  {
    if constexpr (std::is_same_v<T, value_type_of_t<Arg>>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (fixed<Arg>)
    {
#ifdef __cpp_concepts
      if constexpr (requires { typename fixed_value<T, fixed_value_of_v<Arg>>; })
#else
      if constexpr (detail::is_fixable<T, Arg>::value)
#endif
        return fixed_value<T, fixed_value_of_v<Arg>>{};
      else
        return fixed_cast<T, std::decay_t<Arg>>{};
    }
    else
    {
      return static_cast<T>(values::to_value_type(std::forward<Arg>(arg)));
    }
  }


  /**
   * \overload
   * \internal
   * \brief Cast a \ref complex value to another \ref complex value having a particular underlying real value type.
   * \details The result will be complex. If the argument is fixed, the result may or may not be fixed.
   * \tparam T The \ref values::number type associated with the result, which may be either real or complex
   * \tparam Arg A \ref complex value
   */
#ifdef __cpp_concepts
  template<number T, complex Arg> requires
    std::same_as<T, std::decay_t<T>> and
    std::convertible_to<real_type_of_t<Arg>, real_type_of_t<T>>
  constexpr complex decltype(auto)
#else
  template<typename T, typename Arg, std::enable_if_t<
    number<T> and
    complex<Arg> and
    std::is_same_v<T, std::decay_t<T>> and
    detail::real_types_convertible<Arg, T>::value, int> = 0>
  constexpr decltype(auto)
#endif
  cast_to(Arg&& arg)
  {
    if constexpr (std::is_same_v<T, value_type_of_t<Arg>>)
      return std::forward<Arg>(arg);
    else
      return internal::make_complex_number<real_type_of_t<T>>(std::forward<Arg>(arg));
  }

}

#endif
