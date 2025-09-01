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
 * \brief Definition for \ref values::integral.
 */

#ifndef OPENKALMAN_VALUES_INTEGRAL_HPP
#define OPENKALMAN_VALUES_INTEGRAL_HPP

#include "basics/basics.hpp"
#include "value.hpp"
#include "values/traits/value_type_of.hpp"

namespace OpenKalman::values
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct reduces_to_integral : std::false_type {};


    template<typename T>
    struct reduces_to_integral<T, std::enable_if_t<std::is_integral_v<value_type_of_t<T>>>>
      : std::true_type {};
  }
#endif


/**
   * \brief T is an integral \ref value.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept integral = value<T> and std::integral<value_type_of_t<T>>;
#else
  template<typename T>
  constexpr bool integral = value<T> and detail::reduces_to_integral<T>::value;
#endif

}

#endif
