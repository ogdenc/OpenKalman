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
 * \brief Definition for \ref value::integral.
 */

#ifndef OPENKALMAN_VALUE_INTEGRAL_HPP
#define OPENKALMAN_VALUE_INTEGRAL_HPP

#include <type_traits>
#ifdef __cpp_concepts
#include <concepts>
#endif
#include "value.hpp"
#include "values/traits/number_type_of_t.hpp"

namespace OpenKalman::value
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct reduces_to_integral : std::false_type {};


    template<typename T>
    struct reduces_to_integral<T, std::enable_if_t<std::is_integral_v<value::number_type_of_t<T>>>>
      : std::true_type {};
  }
#endif


/**
   * \brief T is an index value.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept integral = value::value<T> and std::integral<value::number_type_of_t<T>>;
#else
  template<typename T>
  constexpr bool integral = value::value<T> and detail::reduces_to_integral<T>::value;
#endif

} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUE_INTEGRAL_HPP
