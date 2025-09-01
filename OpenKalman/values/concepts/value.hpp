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
 * \brief Definition for \ref values::value.
 */

#ifndef OPENKALMAN_VALUES_VALUE_HPP
#define OPENKALMAN_VALUES_VALUE_HPP

#include "number.hpp"
#include "values/traits/value_type_of.hpp"

namespace OpenKalman::values
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct value_impl : std::false_type {};


    template<typename T>
    struct value_impl<T, std::enable_if_t<number<typename values::value_type_of<T>::type>>>
      : std::true_type {};
  }
#endif


/**
   * \brief T is a \ref fixed or \ref dynamic value that is reducible to a \ref values::number "number".
   */
  template<typename T>
#ifdef __cpp_concepts
  concept value = number<value_type_of_t<T>>;
#else
  constexpr bool value = detail::value_impl<T>::value;
#endif

}

#endif
