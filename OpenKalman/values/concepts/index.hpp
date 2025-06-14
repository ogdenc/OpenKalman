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
 * \brief Definition for \ref values::index.
 */

#ifndef OPENKALMAN_VALUE_INDEX_HPP
#define OPENKALMAN_VALUE_INDEX_HPP

#include <type_traits>
#include "values/traits/fixed_number_of.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "integral.hpp"

namespace OpenKalman::values
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct number_type_is_unsigned : std::false_type {};


    template<typename T>
    struct number_type_is_unsigned<T, std::enable_if_t<std::is_unsigned_v<values::number_type_of_t<T>>>>
      : std::true_type {};


    template<typename T, typename = void>
    struct fixed_integral_gt_0 : std::false_type {};


    template<typename T>
    struct fixed_integral_gt_0<T, std::enable_if_t<(values::fixed_number_of<T>::value >= 0)>> : std::true_type {};
  }
#endif


/**
   * \brief T is an index value.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept index = values::integral<T> and (std::is_unsigned_v<number_type_of_t<T>> or fixed_number_of<T>::value >= 0);
#else
  template<typename T>
  constexpr bool index = values::integral<T> and (detail::number_type_is_unsigned<T>::value or detail::fixed_integral_gt_0<T>::value);
#endif

} // namespace OpenKalman::values

#endif //OPENKALMAN_VALUE_INDEX_HPP
